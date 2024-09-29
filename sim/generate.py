import os
import sys
DEBUG = hasattr(sys, 'gettrace') and (sys.gettrace() is not None)
if DEBUG:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
OmegaConf.register_new_resolver("eval", eval)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'mpm'))
import mpm as us
from sim.generate.builder import get_scene
from sim.generate.actions import ActionFactory
from sim.generate.ee import get_ee_state
from sim.generate.topology import Topology
from sim.util import dict_str_to_tuple
import taichi as ti
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle


class GenerationWorkspace:

    def __init__(self, cfg: OmegaConf):
        OmegaConf.resolve(cfg)
        self.cfg = dict_str_to_tuple(cfg)

        # get scene
        self.cam = None
        self.scene, self.cam, self.entities, self.planner = get_scene(self.cfg)
        if self.cfg.render:
            ti.set_logging_level(ti.WARN)
            self.font = ImageFont.truetype("Tests/fonts/NotoSans-Regular.ttf", 18)
        # get topology
        if self.cfg.check.horizon > 0:
            self.topology = Topology(self.cfg, self.scene, self.planner, self.entities)
        else:
            self.topology = None
        # get plan
        self.goals = ActionFactory.get_goals(self.cfg.actions)

        # set by reset()
        self.init_state = None
        self.scene_dir = None

        us.logger.info(f'=== Creating sequence {cfg.scene_id} in {cfg.log.base_dir} ===')

    @property
    def output_dir(self):
        return HydraConfig.get().runtime.output_dir

    def __del__(self):
        if self.cam is not None:
            self.cam.window.destroy()
        us.us_exit()

    def reset(self):
        # get scene dir
        if self.cfg.log.base_dir is None:
            us.logger.error('No log.base_dir specified. Adapt the config file.')
            sys.exit(-1)
        elif not os.path.exists(self.cfg.log.base_dir):
            us.logger.error(f'log.base_dir {self.cfg.log.base_dir} does not exist. Create it or adapt the config file.')
            sys.exit(-1)
        if self.cfg.scene_id is None:
            us.logger.error('No scene_id specified. Adapt the config file.')
            sys.exit(-1)
        self.scene_dir = os.path.join(self.cfg.log.base_dir, self.cfg.scene_id)
        if os.path.exists(self.scene_dir):
            us.logger.warning(f'Scene directory {self.scene_dir} already exists. Overwriting.')
        # save resolved config, including overrides and evals
        os.makedirs(self.scene_dir, exist_ok=True)
        OmegaConf.save(self.cfg, os.path.join(self.scene_dir, 'config.yaml'))
        
        # reset simulation, ee and its state machine
        #   optional: let the scene settle and get the initial state for future resets
        self.planner.reset()
        if self.init_state is None:
            if not DEBUG:
                # let the scene settle
                us.logger.debug('Settling...')
                for _ in range(self.cfg.warmup_horizon):
                    self.scene.step()
                us.logger.debug('Scene settled.')
            self.init_state = self.scene.get_state()
        self.scene.reset(self.init_state)

        # reset topology
        if self.topology is not None:
            self.topology.reset()

    def render(self, step=-1, info_str=''):
        # render state, add id and time, save visualization per subgoal
        frame = Image.fromarray(self.cam.render())
        offset = self.cam.res[0]//8
        ImageDraw.Draw(frame).text((offset, self.cam.res[1] - offset), f"step {step}, {step*self.cfg.sim.step_dt:0.1f}s",
                                   fill='black', anchor='ls', font=self.font)
        if info_str != '':
            ImageDraw.Draw(frame).text((offset, offset), info_str,
                                       fill='black', anchor='lt', font=self.font)
        return frame

    def log(self, step, collision_info, ee_state):
        frame = {
            'step': step,
            't': step * self.scene.sim.step_dt,
            'frame_idx': step//self.cfg.log.n_iter,
        }
        # get particle info
        num_particles = 0
        particles_pos, particles_vel = [], []
        for entity in self.entities:
            if entity.name == 'ground':
                continue
            num_particles += entity.n
            entity_state = entity.get_state()
            particles_pos += [entity_state.pos.detach().cpu().numpy()]
            particles_vel += [entity_state.vel.detach().cpu().numpy()]
        particles_pos = np.concatenate(particles_pos, axis=0)
        particles_vel = np.concatenate(particles_vel, axis=0)
        if self.topology is not None:
            particles_idx = self.topology.particle_graph.entity_indices.cpu().numpy()  # component label
        else:
            particles_idx = np.zeros(num_particles, dtype=us.ITYPE_NP)  # assume single component
        # get ee collision info
        particles_sdf = collision_info[:, 0]  # relative to ee, clipped at 1.0
        particles_colliding = collision_info[:, 2]  # with ee
        
        # compose log frame
        frame['obj'] = {
            'particles': {
                'num': num_particles,
                'pos': particles_pos,
                'vel': particles_vel,
                'idx': particles_idx,
                'sdf': particles_sdf,
                'colliding': particles_colliding,
            },
        }
        if self.topology is not None:
            frame['topology'] = {
                'components': self.topology.get_num_components(),
                'genus': self.topology.get_genus_per_component(),
            }
        frame['ee'] = get_ee_state(self.planner.ee, ee_state)

        return frame

    def check_and_log(self, step):
        if (step*self.cfg.sim.step_dt) % 1 < self.cfg.sim.step_dt:
            us.logger.info(f'current sim keyframe {step//self.cfg.log.n_iter:05d} - elapsed sim time {step*self.cfg.sim.step_dt:0.1f}s')

        # only log simulation keyframes (i.e., every n_iter steps)
        if step % self.cfg.log.n_iter != 0:
            return None, None
        
        # get current collision info -- contains sdf, influence, is colliding
        collision_info = self.scene.sim.collision_info()
        num_colliding = int(collision_info[:, 2].sum())
        if num_colliding > 0 and step == 0:
            us.logger.error(f'ee in collision at init ({num_colliding} particles in collision) - check ee opening and/or object scale')
            sys.exit(-1)
        # render keyframe
        if not DEBUG and self.cfg.render:
            info_str = f'frame {step//self.cfg.log.n_iter}'
            if self.topology is not None:
                info_str = f'{info_str} - {str(self.topology)}'
            keyframe = self.render(step, info_str=info_str)
        else:
            keyframe = None
        # check topology in keyframe
        if num_colliding > 0 and self.topology is not None:
            self.topology.check()

        # log keyframe
        cur_ee_state = self.planner.ee.get_state()
        log = self.log(step, collision_info, cur_ee_state)

        return keyframe, log

    def run(self):
        # reset simulation and state machine
        self.reset()

        logs = []
        frames = []
        self.goals[-1]['wait_after'] = self.cfg.log.wait_after_done
        for step in range(self.cfg.max_horizon):
            # if plan is not done, step state machine and set new ee velocities
            if self.planner.step(self.goals):
                break
            # step simulation
            self.scene.step()

            # check and log step
            keyframe, log = self.check_and_log(step)
            if keyframe is not None:
                frames += [keyframe]
            if log is not None:
                logs += [log]
        us.logger.info(f'=== Done simulating sequence {self.cfg.scene_id} ===')

        us.logger.info(f'Saving logs.')
        path = os.path.join(self.scene_dir, "log.pkl")
        pickle.dump(logs, open(path, 'wb'))
        if (not DEBUG and self.cfg.render):  # save frames as gif
            us.logger.info(f'Saving visualization.')
            frames[0].save(os.path.join(self.scene_dir, f"visualization.gif"),
                        format="GIF", append_images=frames[1:], save_all=True, loop=0,
                        duration=int(self.cfg.log.dt*1000))


@hydra.main(
    version_base=None,
    config_path='./generate/config', 
    config_name='template')
def main(cfg):
    workspace = GenerationWorkspace(cfg)
    workspace.run()


if  __name__== '__main__':
    main()
