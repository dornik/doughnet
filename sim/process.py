import numpy as np
import os
import open3d as o3d
from scipy.spatial import cKDTree
import pickle
import glob
import time
import datetime
import copy
import hydra
from omegaconf import OmegaConf
import hdf5plugin
import h5py
import multiprocessing
from multiprocessing import current_process
import pymeshfix
from colorama import Fore, Style
import sys
DEBUG = hasattr(sys, 'gettrace') and (sys.gettrace() is not None)
if DEBUG:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # required with taichi
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocess'))
from sim.util import resolve_config, dict_str_to_tuple
import sdftoolbox
from sdftoolbox.dual_strategies import LinearEdgeStrategy, NewtonEdgeStrategy, BisectionEdgeStrategy
from sdftoolbox.dual_strategies import NaiveSurfaceNetVertexStrategy, MidpointVertexStrategy  #, DualContouringVertexStrategy
# sdftoolbox has a bug in DualContouringVertexStrategy (non-active voxels have nan values that are not handled correctly)
# -> use our version instead
from sim.process.dual_isosurface import DualContouringVertexStrategy
import logging
logging.getLogger("sdftoolbox").setLevel(logging.WARNING)  # suppress infos for every generated mesh


def particles_to_sdf(config, particles, ee_raycasting=None, in_ee=None):
    valid_particles = particles[~in_ee] if in_ee is not None else particles
    # get particle bbox
    bounds = [valid_particles.min(axis=0) - config.bound_padding, valid_particles.max(axis=0) + config.bound_padding]
    # if this is too small, prune mesh
    if any(bounds[1] - bounds[0] < config.bound_step):  # at least one sample in each dimension
        raise ValueError(f"particle bbox too small: {bounds[1] - bounds[0]}")
    # create grid in particle bbox
    resolutions = [int((bounds[1][i]-bounds[0][i])/config.bound_step) for i in range(3)]
    grid = sdftoolbox.Grid(resolutions, bounds[0], bounds[1])
    # compute NN distances (i.e., to closest particle) for each grid point
    particles_tree = cKDTree(valid_particles)
    samples = grid.xyz
    samples_dist, _ = particles_tree.query(samples.reshape(-1, 3))
    # approximate sign by shifting an outside shell to 0
    #   i.e., particle location is center of a sphere -- shift by radius to get surface
    samples_sdf = samples_dist - config.dist_offset

    if ee_raycasting is not None:
        # avoids that this pushes particles inside the ee mesh
        samples_ee_sdf = ee_raycasting.compute_signed_distance(samples.reshape(-1, 3).astype(np.float32)).numpy()
        samples_sdf[samples_ee_sdf < 0] = config.in_ee_value  # in ee = outside object

    return grid, samples, samples_sdf

def sdf_to_mesh(config, grid, samples=None, samples_sdf=None, sdf=None):
    if sdf is None:
        assert samples is not None and samples_sdf is not None
        # get sdf grid
        samples_sdf = samples_sdf.reshape(samples.shape[:-1])
        sdf = sdftoolbox.sdfs.Discretized(grid, samples_sdf)

    # reconstruct mesh
    # -> Dual Isosurface via Gibson, S. F. F. (1999). Constrained elastic surfacenets: Generating smooth models from binary segmented data.
    if config.vert_strategy == 'naive':
        vertex_strategy = NaiveSurfaceNetVertexStrategy()
    elif config.vert_strategy == 'midpoint':
        vertex_strategy = MidpointVertexStrategy()
    elif config.vert_strategy == 'dual':
        vertex_strategy = DualContouringVertexStrategy()
    else:
        raise ValueError(f"invalid vertex strategy {config.vert_strategy}")
    if config.edge_strategy == 'linear':
        edge_strategy = LinearEdgeStrategy()
    elif config.edge_strategy == 'newton':
        edge_strategy = NewtonEdgeStrategy()
    elif config.edge_strategy == 'bisection':
        edge_strategy = BisectionEdgeStrategy()
    else:
        raise ValueError(f"invalid edge strategy {config.edge_strategy}")
    try:
        mesh_verts, mesh_faces = sdftoolbox.dual_isosurface(sdf, grid, triangulate=True, 
                                                            vertex_relaxation_percent=config.vert_relaxation_percentage,
                                                            vertex_strategy=vertex_strategy,
                                                            edge_strategy=edge_strategy)
        # wrap in open3d mesh and compute normals
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_verts), o3d.utility.Vector3iVector(mesh_faces))
        mesh.compute_vertex_normals()
    except Exception as e:
        print('failed to reconstruct mesh -- probably empty')
        mesh = o3d.geometry.TriangleMesh()
    return mesh

def cleanup_mesh(config, mesh, num_components=1):
    if np.asarray(mesh.vertices).shape[0] == 0:
        return mesh  # NOP for empty

    # filter before
    if config.before.type == 'taubin':
        mesh = mesh.filter_smooth_taubin(config.before.num_iterations)
    elif config.before.type == 'laplacian':
        mesh = mesh.filter_smooth_laplacian(config.before.num_iterations, lambda_filter=config.before.lambda_filter)
    if config.before.type != 'none':
        mesh.compute_vertex_normals()
    # optional: shrink along normal direction
    if config.shrink > 0:
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) - np.asarray(mesh.vertex_normals) * config.shrink)
    # simplify
    mesh = mesh.simplify_quadric_decimation(config.simplify//num_components)
    # filter after
    if config.after.type == 'taubin':
        mesh = mesh.filter_smooth_taubin(config.after.num_iterations)
    elif config.after.type == 'laplacian':
        mesh = mesh.filter_smooth_laplacian(config.after.num_iterations, lambda_filter=config.after.lambda_filter)
    # clean up
    vclean, fclean = pymeshfix.clean_from_arrays(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vclean), o3d.utility.Vector3iVector(fclean))
    #
    mesh.compute_vertex_normals()
    
    return mesh

def consecutive_mesh(mesh, num_target_faces):
    # -- reorder vertices (and labels, i.e., colors) such that face indices are continuous (for batched rendering etc)
    #    i.e., save "face verts" (note: we need the faces bc we may have padding)
    faces = np.arange(num_target_faces*3).reshape(-1, 3)  # assumes faces to be triangular
    verts = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].reshape(-1, 3)
    # -- pad with area-less triangles to get to a fixed number of faces (for batching)
    #    i.e., repeat the same vertex for the superfluous faces
    num_missing_faces = num_target_faces - len(mesh.triangles)
    if num_missing_faces > 0:
        verts = np.vstack([verts, [verts[0]]*num_missing_faces*3])
    if len(verts) != num_target_faces*3:
        raise ValueError(f"expected {num_target_faces*3} verts, got {len(verts)}")
    new_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    # -- add color (if any)
    if len(mesh.vertex_colors) > 0:
        colors = np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)].reshape(-1, 3)
        if num_missing_faces > 0:
            colors = np.vstack([colors, [colors[0]]*num_missing_faces*3])
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return new_mesh, num_missing_faces  # note: can use area == 0 to identify padded faces

def particles_to_mesh(config, particles, ee_raycasting=None, num_components=1, in_ee=None):
    sdf_info = particles_to_sdf(config.sdf, particles, ee_raycasting, in_ee)
    mesh = sdf_to_mesh(config.meshing, *sdf_info)
    mesh = cleanup_mesh(config.cleanup, mesh, num_components=num_components)

    return mesh, sdf_info

def mesh_to_components(mesh):
    # component labels -- convert to face labels (more convenient for point sampling)
    vertex_labels = np.packbits(np.pad(np.asarray(mesh.vertex_colors).astype(np.uint8),
                                       ((0, 0), (5, 0))), axis=1)
    labels = np.unique(vertex_labels)
    assert np.all(labels > 0) and np.all(labels < 8)  # 0 is ee
    face_verts_labels = vertex_labels[np.asarray(mesh.triangles).astype(np.int32)]
    face_masks = [np.all(face_verts_labels == label, axis=1) for label in labels]
    assert np.sum(face_masks) == len(mesh.triangles)  # each face must have one label
    face_labels = labels[np.argmax(face_masks, axis=0)].astype(np.int32)
    component_meshes = []
    for label in np.unique(face_labels):
        indices = np.argwhere(vertex_labels.squeeze() == label).squeeze()
        component_mesh = mesh.select_by_index(indices, cleanup=True)
        component_meshes += [component_mesh]
    return component_meshes, vertex_labels, face_labels

def load_ee_mesh(mesh_config):
    meshes_dir = os.path.join(os.path.dirname(__file__), 'mpm/assets/meshes/processed')
    mesh_path = mesh_config.geom.file
    # note: meshes preprocessed for simulator are renamed to [name]-[name].obj
    mesh = o3d.io.read_triangle_mesh(os.path.join(meshes_dir, f"{mesh_path.split('.obj')[0]}-{mesh_path}"))
    # to object frame
    mesh.transform(np.diag(mesh_config.geom.scale + [1.0]))
    mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.deg2rad(mesh_config.geom.offset_euler)), np.zeros((3, 1)))
    mesh.translate(mesh_config.geom.offset_pos)
    return mesh

def get_ee_in_frame(ee_meshes, ee_sample):
    assert len(ee_meshes) in [1, 2]
    ee_fingers_frame = [copy.deepcopy(mesh) for mesh in ee_meshes]
    finger_suffix = ['_left', '_right'] if len(ee_meshes) == 2 else ['']
    for k, mesh in zip(finger_suffix, ee_fingers_frame):
        mesh.rotate(mesh.get_rotation_matrix_from_quaternion(ee_sample['quat']), np.zeros((3, 1)))
        mesh.translate(ee_sample[f'pos{k}'])
    if len(ee_fingers_frame) == 2:  # merge meshes
        ee_frame = ee_fingers_frame[0] + ee_fingers_frame[1]
    else:
        ee_frame = ee_fingers_frame[0]
    return ee_frame, ee_fingers_frame

def get_ee_samples(ee_fingers, zmax, num_samples):
    # enforce zmax
    mesh_ee_left, mesh_ee_right = ee_fingers
    transform = np.eye(4)
    transform[2, 2] = zmax / mesh_ee_left.get_max_bound()[2]  # assume same for both
    mesh_ee_left.transform(transform)
    mesh_ee_right.transform(transform)
    # sample same number of points from each finger
    samples_left = np.asarray(mesh_ee_left.sample_points_uniformly(number_of_points=num_samples//2).points).astype(np.float32)
    samples_right = np.asarray(mesh_ee_right.sample_points_uniformly(number_of_points=num_samples//2).points).astype(np.float32)
    samples = np.concatenate([samples_left, samples_right], axis=0)
    return samples

def mesh_to_sample(mesh, prefix='obj_'):
    return {
        f'{prefix}verts': np.asarray(mesh.vertices).astype(np.float32),
        f'{prefix}faces': np.asarray(mesh.triangles).astype(np.int32),
    }

def initializer(writer_lock):
    # workaround for multiprocessing with Pool and starmap_async
    #  (e.g., lock is not serializable, hence not possible to add it to writer itself)
    global lock
    lock = writer_lock

def process_sequence(seq_idx, seq_dir, writer, cfg, DEBUG=False, total=-1):
    st = time.time()
    log_prefix = f'worker: {current_process()._identity[0]:02d}, pid: {os.getpid()} ==> ' if not DEBUG else ''
    if 'lock' not in globals():
        lock = None  # single process; no lock used in writer

    # load scene config (used for simulation)
    s_cfg = OmegaConf.load(os.path.join(seq_dir, 'config.yaml'))
    s_cfg = resolve_config(s_cfg)
    print(f"{Fore.BLUE}{log_prefix}processing sequence {seq_idx+1:04d}/{total:04d} (scene_id = {s_cfg.scene_id}); starting at {datetime.datetime.now()}...{Style.RESET_ALL}", flush=True)
    # load generated frames (from simulation)
    generated = pickle.load(open(os.path.join(seq_dir, 'log.pkl'), 'rb'))
    # load ee
    if s_cfg.ee.type in ['gripper',]:
        ee_meshes = [load_ee_mesh(s_cfg.ee.entities.finger_left),
                     load_ee_mesh(s_cfg.ee.entities.finger_right)]
    else:
        ee_meshes = [load_ee_mesh(s_cfg.ee.entities.finger)]

    # iterate over frames
    time_elapsed = 0
    previous_mesh = None
    was_inee = None
    num_colliding = np.array([g['obj']['particles']['colliding'].sum() for g in generated[-cfg.num_frames:]]).sum()  # for all frames
    for frame_id, sample in enumerate(generated[-cfg.num_frames:]):  # only last cfg.num_frames (padded afterwards if fewer generated)

        if (time.time() - st) - time_elapsed > 60:
            time_elapsed += 60
            print(f"   {log_prefix}processing frame {frame_id+1}/{min(cfg.num_frames, len(generated))} in sequence {seq_idx+1} (t={int(time_elapsed)}s)...", flush=True)

        # -- extract object and ee meshes (incl component labels)

        # get ee mesh in current frame
        ee_frame, ee_fingers_frame = get_ee_in_frame(ee_meshes, sample['ee'])

        # speed up: compute scene mesh once and skip recomputation while no scene particles are colliding with the ee
        if num_colliding > 0 or previous_mesh is None:

            # create raycasting scene to compute sdf wrt ee
            ee_raycasting = o3d.t.geometry.RaycastingScene()
            _ = ee_raycasting.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ee_frame))

            # get particles in current frame
            particles = sample['obj']['particles']['pos']
            
            # === compute mesh per component

            # - initial meshes
            scene_meshes = []
            scene_sdf_infos = []
            
            # --
            particle_labels = sample['obj']['particles']['idx']
            assert np.max(particle_labels) < 5
            # relabel to get number of parts down
            #   if 0 and 1 merge (= new component is 2) and then split (= two new components are {3,4}),
            #   we can safely relabel 4 to 0
            particle_labels[particle_labels == 4] = 0
            particle_components = np.unique(particle_labels)
            num_components = particle_components.shape[0]
            if 'topology' in sample:
                assert num_components == sample['topology']['components']

            particles_inee = ee_raycasting.compute_signed_distance(particles.astype(np.float32)).numpy() < 0.001
            # avoid pruned particles from reappearing later for consistency
            if was_inee is None:
                was_inee = np.zeros(particles.shape[0], dtype=bool)
            was_inee[particles_inee] = True

            for component_label in particle_components:
                component_particles = particles[particle_labels == component_label]
                mesh, sdf_info = particles_to_mesh(cfg, component_particles, ee_raycasting, num_components=num_components, in_ee=was_inee[particle_labels == component_label])
                # encode label in color; note: 0 reserved for ee (hence +1)
                component_color = np.unpackbits(np.array(component_label + 1).astype(np.uint8))[-3:]
                mesh.paint_uniform_color(component_color)
                
                scene_meshes += [mesh]
                scene_sdf_infos += [sdf_info]

            if len(scene_meshes) > 1:

                # - check overlap/intersection between components
                scene_mesh = o3d.geometry.TriangleMesh()
                for mi, (component_mesh, (component_grid, component_samples, component_sdf)) in enumerate(zip(scene_meshes, scene_sdf_infos)):
                    # one raycasting scene with all other meshes
                    other_meshes = o3d.geometry.TriangleMesh()
                    for m in scene_meshes[:mi] + scene_meshes[mi+1:]:
                        other_meshes += m
                    raycasting_scene = o3d.t.geometry.RaycastingScene()
                    _ = raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(other_meshes))
                    # check intersection
                    component_intersects = raycasting_scene.compute_signed_distance(component_samples.reshape(-1, 3).astype(np.float32)).numpy() < 0
                    # set sdf to slightly 'outside' (i.e., >0) for intersecting samples
                    component_sdf[component_intersects] = 0.00001
                    # reconstruct mesh with updated sdf
                    new_component_mesh = sdf_to_mesh(cfg.meshing, component_grid, component_samples, component_sdf)
                    new_component_mesh = cleanup_mesh(cfg.cleanup, new_component_mesh, num_components=len(scene_meshes))
                    new_component_mesh.paint_uniform_color(component_mesh.vertex_colors[0])
                    # merge components into single mesh
                    scene_mesh += new_component_mesh
                mesh = scene_mesh
            else:
                mesh = scene_meshes[0]

            # no particles in scene; no meshes
            if len(mesh.vertices) == 0:
                print(f"== {log_prefix}no particles in scene, stopping at frame{frame_id} - check for errors")
                break

            # -- reorder vertices (and labels, i.e., colors) such that face indices are continuous (for batched rendering etc)
            # i.e., save "face verts" (note: we need the faces bc we may have padding)
            try:
                mesh, num_dummy_faces = consecutive_mesh(mesh, cfg.cleanup.simplify)
            except ValueError as ex:
                print(f"== {log_prefix}failed to generate consecutive obj mesh, stopping at frame{frame_id}\n{ex}")
                break

            # mesh can be re-used if there is no collision (e.g., at the beginning or when objects are missed/static)
            previous_mesh = mesh
        else:
            # keep previous mesh
            mesh = previous_mesh
        
        # also make ee mesh be of constant size and consecutive order (to simplify batching, rendering etc)
        # note: assumes that we treat ee as a single mesh (not, e.g., split into two fingers)
        ee_frame = ee_frame.simplify_quadric_decimation(cfg.ee.num_simplified_faces)
        try:
            ee_frame, num_ee_dummy_faces = consecutive_mesh(ee_frame, cfg.ee.num_simplified_faces)
        except ValueError as ex:
            print(f" == {log_prefix}failed to generate consecutive ee mesh, stopping at frame{frame_id}\n{ex}")
            break

        # -- save train/test samples
        # - object(s)
        component_meshes, vertex_labels, face_labels = mesh_to_components(mesh)
        num_components = len(component_meshes)
        # - ee
        ee_observed = get_ee_samples(ee_fingers_frame, 0.15, 1024)
        # - compose the final preprocessed sample
        item = {
            'scene': int(s_cfg.scene_id), 'frame': frame_id,  # original scene id, (new) consecutive frame id
            # component labels
            'obj_vert_labels': vertex_labels.astype(np.int32),
            'obj_face_labels': face_labels.astype(np.int32),
            # ee
            'ee_observed': ee_observed.astype(np.float32),
        }
        if 'topology' in sample:
            # make topological information collatable
            component_labels = np.pad(particle_components, (0, 5 - len(particle_components)), mode='constant', constant_values=-1)
            genus = np.pad(sample['topology']['genus'], (0, 5 - len(sample['topology']['genus'])), mode='constant', constant_values=-1)
            # correctly assign genus to component; merge information into single vector
            comp_labels = component_labels + 1  # the index (label on mesh) where the genus actually belongs to
            comp_genus = -np.ones_like(genus)  # -1 indicates 'no component', >=0 the component's genus
            for g, l in zip(genus, comp_labels):
                comp_genus[l] = g
            item['genus'] = comp_genus
        item.update(mesh_to_sample(mesh))
        item.update(mesh_to_sample(ee_frame, prefix='ee_'))
        # - save
        writer.write(lock, item, seq_idx, frame_id)

    # too few frames? pad same at end
    if frame_id + 1 < cfg.num_frames:
        print(f"== {log_prefix}padding sequence {seq_idx+1:04d} with {cfg.num_frames - frame_id - 1} frames")
        for frame_id in range(frame_id + 1, cfg.num_frames):
            item['frame'] = frame_id
            writer.write(lock, item, seq_idx, frame_id)

    print(f'{Fore.GREEN}{log_prefix} done with {seq_idx+1:04d} after {int(time.time()-st)}s{Style.RESET_ALL}')


class Writer:
    def __init__(self, base_dir, num_scenes, num_frames):
        self.base_dir = base_dir
        self.num_scenes = num_scenes
        self.num_frames = num_frames

    def write(self, lock, frame_data, scene_idx, frame_idx):
        assert 0 <= scene_idx < self.num_scenes
        assert 0 <= frame_idx < self.num_frames

        if lock is not None:
            lock.acquire()

        with h5py.File(os.path.join(self.base_dir, 'data.h5'), 'a') as f:
            for k, v in frame_data.items():
                # create dataset (if not exists)
                if isinstance(v, np.ndarray):
                    ds_shape = (self.num_scenes, self.num_frames, *v.shape)
                    ds_dtype = v.dtype
                else:
                    ds_shape = (self.num_scenes, self.num_frames)
                    ds_dtype = np.dtype(type(v))
                if ds_dtype in [np.int64, 'int64']:
                    ds_dtype = np.int32
                elif ds_dtype in [np.float64, 'float64']:
                    ds_dtype = np.float32
                d = f.require_dataset(k, shape=ds_shape, dtype=ds_dtype, exact=True,
                                        chunks=(1, 1, *ds_shape[2:]),
                                        compression=hdf5plugin.Blosc2(),)
                # write data
                d[scene_idx, frame_idx, ...] = v
        
        if lock is not None:
            lock.release()


@hydra.main(
    version_base=None,
    config_path='./process/config', 
    config_name='common')
def main(cfg):
    OmegaConf.resolve(cfg)
    cfg = dict_str_to_tuple(cfg)

    # get all sequences in the base directory
    base_dir = cfg.base_dir
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"root directory {base_dir} not found - check config file")
    seq_names = sorted(glob.glob(os.path.join(base_dir, '*')))
    seq_names = [d for d in seq_names
                if os.path.isdir(d)
                and os.path.exists(os.path.join(d, 'log.pkl'))  # has generated data
                and not os.path.exists(os.path.join(d, 'data.h5'))  # not yet processed
    ]
    if len(seq_names) == 0:
        print(f"no remaining sequences to process in {base_dir}")
        return
    if os.path.exists(os.path.join(base_dir, 'data.h5')):
        out = None
        while out not in ['Y', 'y', '', 'n']:
            out = input(f"File {os.path.join(base_dir, 'data.h5')} already exists. Replace? [Y/n]")
        if out == 'n':
            return
        os.remove(os.path.join(base_dir, 'data.h5'))

    writer = Writer(base_dir, len(seq_names), cfg.num_frames)

    if DEBUG or cfg.num_processes == 1:  # single process

        for seq_idx, seq_name in enumerate(seq_names):
            process_sequence(seq_idx, seq_name, writer, cfg, True)

    else:  # multiprocessing

        def error_callback(error):
            print(f'{Fore.RED}error: {error}{Style.RESET_ALL}')

        multiprocessing.set_start_method('forkserver' if not sys.platform == 'win32' else 'spawn', force=True)
        with multiprocessing.Pool(
                processes=int(min(len(seq_names), cfg.num_processes)),
                initializer=initializer, initargs=(multiprocessing.Lock(),)
                ) as pool:
            result = pool.starmap_async(
                process_sequence,
                [(seq_idx, seq_name, writer, cfg, False, len(seq_names)) for seq_idx, seq_name in enumerate(seq_names)],
                error_callback=error_callback,
            )
            result.get()
            pool.close()
            pool.join()


if  __name__== '__main__':
    main()
