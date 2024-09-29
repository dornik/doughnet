import socket
import os
import mpm as us

from sim.generate.ee import EndEffector
from sim.generate.state_machine import StateMachine


def get_material(scene, config):
    if config.type == 'tool':
        return us.materials.Tool(
            scene=scene, name=config.name,
            friction=config.friction, contact_softness=config.contact_softness,
            collision=True, collision_type=config.collision_type,
            sdf_res=128,
        )
    elif config.type == 'elastoplastic':
        return us.materials.ElastoPlastic(
            scene=scene, name=config.name,
            mu=config.lame_mu, lam=config.lame_lambda, rho=config.rho,
            yield_lower=config.yield_low_high[0], yield_higher=config.yield_low_high[0],
            filling=config.filling,
        )
    else:
        raise ValueError(f"unknown material type {config.type}")
    
def get_geom(config, name):
    if config.type == 'cylinder':
        return us.geoms.Cylinder(
            center=config.center, height=config.height, radius=config.radius, euler=config.euler, name=name,
        )
    elif config.type == 'cube':
        return us.geoms.Cube(
            lower=config.lower, upper=config.upper, euler=config.euler, name=name,
        )
    elif config.type == 'supertoroid':
        return us.geoms.Supertoroid(
            center=config.center, size=config.size, hole=config.hole,
            e_lat=config.e_lat, e_lon=config.e_lon, euler=config.euler, name=name,
        )
    elif config.type == 'mesh':
        return us.geoms.Mesh(
            file=config.file, offset_pos=config.offset_pos, offset_euler=config.offset_euler,
            pos=config.pos, euler=config.euler, scale=config.scale, name=name,
        )
    else:
        raise ValueError(f"unknown geom type {config.type}")

def get_entity(scene, config, name):
    return scene.add_entity(
        material=get_material(scene, config.material),
        geom=get_geom(config.geom, name),
        surface_options=us.options.SurfaceOptions(
            color=config.surface.color,
        )
    )

def get_ee(scene, config):
    entities = [get_entity(scene, entity_config, name) for name, entity_config in config.entities.items()]
    ee = StateMachine(EndEffector(config, entities))
    return ee

def get_scene(config):

    # initialize simulation
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_id)
    os.environ['TI_VISIBLE_DEVICE'] = str(config.cuda_id)
    us.init(seed=config.sim.seed, allocate_gpu_memory=config.sim.allocate_gpu_memory,
            precision=str(config.sim.precision), logging_level=config.sim.logging_level)

    # define scene and solver
    scene = us.Scene(
        sim_options=us.options.SimOptions(
            max_substeps_local=config.sim.max_substeps_local,
            gravity=config.sim.gravity,
        ),
        tool_options=us.options.ToolOptions(
            step_dt=config.sim.step_dt,  # overriden by defaults in SimOptions if not set
            substep_dt=config.sim.substep_dt,
            floor_height=config.sim.tool.floor_height,
        ),
        mpm_options=us.options.MPMOptions(
            step_dt=config.sim.step_dt,  # overriden by defaults in SimOptions if not set
            substep_dt=config.sim.substep_dt,
            particle_diameter=config.sim.mpm.particle_diameter,
            grid_density=config.sim.mpm.grid_density,
            # note: boundary is padded by 3*dx = 3/grid_density = 3/64 by default
            lower_bound=config.sim.mpm.lower_bound,
            upper_bound=config.sim.mpm.upper_bound,
        ),
        viewer_options=us.options.ViewerOptions(),
    )
    
    # optional: add camera
    if config.render:
        cam = scene.add_camera(
            res=(config.cam.width, config.cam.height), fov=config.cam.fov,
            pos=config.cam.pos, lookat=config.cam.lookat, up=config.cam.up,
        )
    else:
        cam = None

    # scene objects
    entities = [get_entity(scene, entity_config, name) for name, entity_config in config.entities.items()]

    # ee objects with controller, wrapped in a state machine
    planner = get_ee(scene, config.ee)

    # build scene
    scene.build()
    scene.reset()

    return scene, cam, entities, planner
