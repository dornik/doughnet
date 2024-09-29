import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import knn_graph, approx_knn_graph, radius_graph
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
from scipy.sparse import csgraph
import open3d as o3d

import mpm as us
from mpm.utils.geom import euler_to_quat
from sim.generate.ee import transform_by_quat, quat_to_R


class SceneGraph:

    """
    Starts from the objects' initial topology in terms of a simple planar graph. 
    Each recorded change alters this graph using merge, split, and remove graph operations. 
    This scene-topology graph is easily evaluated in terms of the number of connected components 
    and cycles, where the number of cycles is equivalent to the genus of the corresponding object.
    """

    def __init__(self, scene_type) -> None:
        self.scene_type = scene_type

        # define the initial planar graph per supported scene type...
        if scene_type == 'g0':  # c=1, g={0}
            #    V     finger V
            #  1-0-2   0 is where ee will have its effect
            #    A     finger A
            nodes = list(range(3))
            edges = [(0, 1),  # top
                     (0, 2),]  # bottom
            comp  = [0]*5
            side  = ['mid', 'above', 'below',]  # relative to ee plane
            kind  = ['merge', 'split', 'split',]
            pos   = [[0, 0], [0, 1], [0, -1],]
        elif scene_type == 'g0g0':  # c=2, g={0,0}
            #    V     finger V
            #  1-0-2   0 is where finger V will have its effect
            #  =====   gap between components
            #  5-3-4   3 is where finger A will have its effect
            #    A     right finger
            nodes = list(range(6))
            edges = [(0, 1),  # top, first component
                     (0, 2),  # bottom, first component
                     (3, 4),  # bottom, second component
                     (3, 5),]  # top, second component
            comp  = [0]*3 + [1]*3
            side  = ['mid', 'above', 'below', 'mid', 'below', 'above',]
            kind  = ['merge', 'split', 'split', 'merge', 'split', 'split',]
            pos   = [[0, 0], [0, 1], [0, -1],
                     [1, 0], [1, -1], [1, 1],]
        elif scene_type == 'g1':  # c=1, g={1}
            #      V       finger V
            #    1-0-2     0 is where finger V will have its effect
            #  /       \
            # 3         4  boundary between potential first component above and second below
            #  \       /
            #    7-5-6     5 is where finger A will have its effect
            #      A       finger A
            nodes = list(range(8))
            edges = [(0, 1), (1, 3),  # top, first component
                     (0, 2), (2, 4),  # bottom, first component
                     (5, 6), (6, 4),  # bottom, second component
                     (5, 7), (7, 3)]  # top, second component
            comp  = [0]*8
            side  = ['mid', 'above', 'below', 'above', 'below', 'mid', 'below', 'above']
            kind  = ['merge', 'split', 'split', 'end', 'end', 'merge', 'split', 'split']
            pos   = [[0, 0], [0, 1], [0, -1],
                     [0.5, 2], [0.5, -2],
                     [1, 0], [1, -1], [1, 1]]
        else:
            raise NotImplementedError(f'Unknown scene type {scene_type}')
        
        # unidirected graph; self-loops are allowed, but not parallel edges
        self.graph = nx.Graph()
        # create corresponding graph
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        # add attributes
        nx.set_node_attributes(self.graph, dict(zip(nodes, comp)), name='comp')
        nx.set_node_attributes(self.graph, dict(zip(nodes, side)), name='side')
        nx.set_node_attributes(self.graph, dict(zip(nodes, kind)), name='kind')
        nx.set_node_attributes(self.graph, dict(zip(nodes, pos)), name='pos')

        # helper flags
        self.new_node_id = len(self)
        self.is_merged = scene_type == 'g0'  # c=1; no additional merging possible
        self.is_split = False

    def __len__(self):
        return self.graph.number_of_nodes()
    
    def get_genus_per_component(self):
        return [len(list(nx.simple_cycles(self.graph.subgraph(component))))
                for component in nx.connected_components(self.graph)]
    
    def get_num_components(self):
        return nx.number_connected_components(self.graph)

    def get_entity_nodes(self, entity_idx):
        return [n for n in self.graph.nodes()
                if self.graph.nodes[n]['comp'] == entity_idx]

    def split(self, node):
        assert self.graph.nodes[node]['kind'] == 'merge' and self.graph.nodes[node]['side'] == 'mid'
        old_pos = self.graph.nodes[node]['pos']
        old_comps = [self.graph.nodes[n]['comp'] for n in self.graph.nodes]
        # get all nodes above and below
        nodes_above = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == 'above']
        nodes_below = [n for n in self.graph.nodes() if self.graph.nodes[n]['side'] == 'below']
        # get edges with {node}
        edges_above = [list(edge) for edge in self.graph.edges(nodes_above) if node in edge]
        edges_below = [list(edge) for edge in self.graph.edges(nodes_below) if node in edge]
        # remove node and insert two new nodes
        new_node_above = self.new_node_id
        new_node_below = self.new_node_id + 1
        self.new_node_id += 2
        self.graph.remove_node(node)
        self.graph.add_nodes_from([new_node_above, new_node_below])
        # update edges
        for edge in edges_above:
            if edge[0] == node:
                edge[0] = new_node_above
            else:
                edge[1] = new_node_above
        for edge in edges_below:
            if edge[0] == node:
                edge[0] = new_node_below
            else:
                edge[1] = new_node_below
        self.graph.add_edges_from(edges_above + edges_below)
        # update attributes
        new_components = nx.connected_components(self.graph)
        for component in new_components:
            if new_node_above in component:
                for node in component:
                    self.graph.nodes[node]['comp'] = np.max(old_comps) + 1
            if new_node_below in component:
                for node in component:
                    self.graph.nodes[node]['comp'] = np.max(old_comps) + 2
        assert np.unique([self.graph.nodes[n]['comp']  # check that we have the right number of components
                          for n in self.graph.nodes()]).shape[0] == len(list(nx.connected_components(self.graph)))

        self.graph.nodes[new_node_above]['side'] = 'above'
        self.graph.nodes[new_node_below]['side'] = 'below'
        self.graph.nodes[new_node_above]['kind'] = 'end'
        self.graph.nodes[new_node_below]['kind'] = 'end'
        self.graph.nodes[new_node_above]['pos'] = [old_pos[0], old_pos[1] + 0.5]
        self.graph.nodes[new_node_below]['pos'] = [old_pos[0], old_pos[1] - 0.5]
        # update helper flags
        self.is_split = True
        return new_node_above, new_node_below

    def merge(self, side, entity_idx=None):
        # get all nodes (of a specific entity)
        nodes = self.graph.nodes() if entity_idx is None else self.get_entity_nodes(entity_idx)
        # get all nodes on {side}
        nodes = [n for n in nodes if self.graph.nodes[n]['side'] == side]

        old_sides = [self.graph.nodes[n]['side'] for n in nodes]
        old_kinds = [self.graph.nodes[n]['kind'] for n in nodes]
        old_pos   = [self.graph.nodes[n]['pos'] for n in nodes]
        old_comps = [self.graph.nodes[n]['comp'] for n in nodes]
        old_components = list(nx.connected_components(self.graph))
        # update nodes and edges (automatically merged by networkx)
        new_node = self.new_node_id
        self.new_node_id += 1
        nx.relabel_nodes(self.graph, {node: new_node for node in nodes}, copy=False)
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))
        # update attributes
        for component in old_components:
            if np.any([n in component for n in nodes]):
                for node in component:
                    if node in nodes:
                        self.graph.nodes[new_node]['comp'] = np.max(old_comps) + 1
                    else:
                        self.graph.nodes[node]['comp'] = np.max(old_comps) + 1
        assert np.unique([self.graph.nodes[n]['comp']  # check that we have the right number of components
                          for n in self.graph.nodes()]).shape[0] == len(list(nx.connected_components(self.graph)))
        assert np.all([side == old_sides[0] for side in old_sides])
        self.graph.nodes[new_node]['side'] = old_sides[0]
        self.graph.nodes[new_node]['kind'] = old_kinds[0]  # assumes that they are given in the right order
        self.graph.nodes[new_node]['pos'] = np.mean(old_pos, axis=0)
        # update helper flags
        self.is_merged = True
        return new_node

    def remove(self, side, entity_idx=None):
        # get all nodes (of a specific entity)
        nodes = self.graph.nodes() if entity_idx is None else self.get_entity_nodes(entity_idx)
        # get all nodes on {side}
        nodes = [n for n in nodes if self.graph.nodes[n]['side'] == side]
        self.graph.remove_nodes_from(nodes)
        return nodes

    def draw(self, attribute=None, pos=None):
        if pos is None:
            pos = {n: self.graph.nodes[n]['pos'] for n in self.graph.nodes()}

        if attribute is None:
            colors = '#1f78b4'
        elif attribute in ['comp', 'side', 'kind']:
            if attribute == 'comp':
                palette = {0: 'red', 1: 'blue', 2: 'green', 3: 'magenta', 4: 'cyan'}
            elif attribute == 'side':
                palette = {'mid': 'black', 'above': 'green', 'below': 'orange'}
            else:  # kind
                palette = {'merge': 'green', 'split': 'orange', 'end': 'blue'}
            colors = [palette[self.graph.nodes[n][attribute]] for n in self.graph.nodes()]
        else:
            raise NotImplementedError(f'Unknown attribute {attribute}')
        nx.draw_networkx(self.graph, pos=pos, node_color=colors, with_labels=True)
        return pos
    
    def draw_all(self, show=True, title_prefix=''):
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        pos = self.draw('comp')
        plt.title('comp')
        plt.subplot(1, 2, 2)
        self.draw('side', pos)
        plt.title('side')
        plt.suptitle(f'{title_prefix}{self.scene_type} scene with {self.get_num_components()} component(s) of genus {self.get_genus_per_component()}')
        if show:
            plt.show()
    
    def save_graph_visualization(self, path, title_prefix=''):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.clf()
        self.draw_all(show=False, title_prefix=title_prefix)
        plt.savefig(path)


class ParticleGraph:

    """
    Starts from the particles' initial positions and their neighborhood graph.
    """

    def __init__(self, scene_type, entities):
        self.scene_type = scene_type

        # current state
        particles_at_rest, entity_particle_counts = get_scene_particles(entities)
        # pos
        self.particles_at_rest = particles_at_rest
        # neighborhood - initially knn to avoid issues with density
        self.particle_graph = self._get_neighborhood_graph(self.particles_at_rest, mode='knn')
        # entity
        num_components = len(entity_particle_counts)
        self.entity_particle_counts = entity_particle_counts
        self.entity_indices = torch.zeros(self.particles_at_rest.shape[0],
                                          dtype=torch.long, device=self.particles_at_rest.device)
        for entity_idx in range(num_components):
            start_idx = int(np.cumsum(self.entity_particle_counts[:entity_idx])) if entity_idx > 0 else 0
            end_idx = int(np.cumsum(self.entity_particle_counts[:entity_idx+1])[-1])
            self.entity_indices[start_idx:end_idx] = entity_idx
        # helper flags
        self.is_missed = np.array([False]*num_components)  # per entity
        self.is_ripped = np.array([False]*num_components)  # per entity

    def check_missed(self, particles, ee_pose, ee_width):
        inhull_labels = self.get_between_region(particles, ee_pose, ee_width, 0.5)
        entity_labels = self.entity_indices
        #
        num_components = len(self.entity_particle_counts)
        self.is_missed = np.zeros(num_components, dtype=bool)
        for ei, entity_idx in enumerate(torch.unique(self.entity_indices)):
            entity_mask = entity_labels == entity_idx
            entity_inhull = inhull_labels[entity_mask]
            self.is_missed[ei] = entity_inhull.sum() == 0  # no particle between fingers
        return np.all(self.is_missed)

    def check_ripped(self, particles, r=0.02):
        if self.scene_type == 'g1g1' and not np.all(self.is_ripped):
            num_components = len(self.entity_particle_counts)
            self.is_ripped = np.zeros(num_components, dtype=bool)
            for ei, entity_idx in enumerate(torch.unique(self.entity_indices)):
                entity_mask = self.entity_indices == entity_idx
                entity_particles = particles[entity_mask]
                entity_graph = self._get_neighborhood_graph(entity_particles, r=r)

                entity_components, component_labels = self._get_particle_labels(entity_graph, entity_particles.shape[0],
                                                                                ordered_by='ordered_by')
                self.is_ripped[ei] = entity_components > 1
            return np.any(self.is_ripped)  # either is not connected
        return False

    def check_merge_before(self, particles, ee_pose, ee_width, ee_open):
        # get particles in ee frame
        particles_in_ee_frame = torch.matmul(particles - ee_pose[:3, 3], ee_pose[:3, :3])
        # get labels based on current particle positions and existing neighborhood graph
        slice_thresholds = self._get_slice_thresholds(particles_in_ee_frame)
        particle_labels = -torch.ones_like(particles_in_ee_frame[:, 0])
        particle_labels, _ = self._get_particle_labels_sliced(particles_in_ee_frame, particle_labels,
                                                              self.particle_graph, slice_thresholds)
        
        # - check if the two regions are touching already (-> requires sim-based check)
        merge_mask = torch.logical_or(particle_labels == 0, particle_labels == 1)
        merge_particles = particles[merge_mask]
        # check if (merge & between) are connected wrt distance
        between_mask = self.get_between_region(merge_particles, ee_pose, ee_width, ee_open)
        check_graph = self._get_neighborhood_graph(merge_particles[between_mask == 1])
        check_components, check_labels = self._get_particle_labels(check_graph, (between_mask == 1).sum())
        needs_check = check_components == 1

        return particle_labels, needs_check
    
    def check_merge_after(self, particles_check, ee_pose, ee_width, ee_open, merge_mask, r=0.03):
        # get particles strictly -between- fingers wrt the original opening
        inhull_labels = self.get_between_region(particles_check, ee_pose, ee_width, 0.5)
        between_labels = self.get_between_region(particles_check, ee_pose, ee_width, ee_open)
        inee_mask = torch.logical_and(inhull_labels == 1, between_labels == 0)
        between_mask = torch.logical_and(~inee_mask, between_labels == 1)

        # check if those particles are connected wrt distance
        merge_between_mask = torch.logical_and(merge_mask, between_mask)
        if merge_between_mask.sum() == 0:
            return False
        
        # get particles in ee frame
        particles_in_ee_frame = torch.matmul(particles_check - ee_pose[:3, 3], ee_pose[:3, :3])

        check_particles = particles_in_ee_frame[merge_between_mask]
        check_graph = self._get_neighborhood_graph(check_particles, r=r)
        slice_thresholds = self._get_slice_thresholds(check_particles)
        particle_labels = -torch.ones_like(check_particles[:, 0])
        particle_labels, per_slice_components = self._get_particle_labels_sliced(check_particles,  
                                                                                 particle_labels,
                                                                                 check_graph, slice_thresholds,
                                                                                 ordered_by='size')

        min_merged_slices = np.ceil(per_slice_components.shape[0] * 0.5)  # at least half
        return np.sum(per_slice_components <= 1) >= min_merged_slices, inhull_labels
    
    def merge(self, particles_sim, inhull_labels, entity_mask, ee_pose, ee_width):
        entity_particles = particles_sim[entity_mask]
        entity_inhull = inhull_labels[entity_mask]
        entity_graph = subgraph(torch.argwhere(entity_mask).view(-1), self.particle_graph, relabel_nodes=True)[0]

        # get regions above an below merge hull
        split_region, _ = self.check_split_before(entity_particles, ee_pose, ee_width)
        above_mask = torch.logical_and(split_region == 1, entity_inhull == 0)  # above and not in hull
        below_mask = torch.logical_and(split_region == 0, entity_inhull == 0)  # below and not in hull

        # check if above/below empty ~now~
        above_empty = above_mask.sum() == 0
        below_empty = below_mask.sum() == 0

        # need to self-merge above?
        if not above_empty and self.scene_type == 'g1':
            above_graph = subgraph(torch.argwhere(above_mask).view(-1), entity_graph, relabel_nodes=True)[0]
            above_connected, _ = self._get_connected(entity_particles[above_mask], above_graph, ee_pose)
        else:
            above_connected = False
        # need to self-merge below?
        if not below_empty and self.scene_type == 'g1':
            below_graph = subgraph(torch.argwhere(below_mask).view(-1), entity_graph, relabel_nodes=True)[0]
            below_connected, _ = self._get_connected(entity_particles[below_mask], below_graph, ee_pose)
        else:
            below_connected = False

        return [above_empty, below_empty], [above_connected, below_connected]

    def check_split_before(self, particles, ee_pose, ee_width, min_cluster_size=0):
        # get particles in ee frame
        particles_in_ee_frame = torch.matmul(particles - ee_pose[:3, 3], ee_pose[:3, :3])
        # check against plane -> top/bottom
        particle_labels = (particles_in_ee_frame[:, 1] > 0).float()

        # only if both sides outside the hull (above/below) are "non-empty" (i.e., have at least min_cluster_size particles)
        inhull_labels = self.get_between_region(particles, ee_pose, ee_width, 0.5)
        above_outside_mask = torch.logical_and(particle_labels == 1, inhull_labels == 0)
        below_outside_mask = torch.logical_and(particle_labels == 0, inhull_labels == 0)
        needs_check = above_outside_mask.sum() > min_cluster_size and below_outside_mask.sum() > min_cluster_size

        return particle_labels, needs_check
    
    def check_split_after(self, particles_check, inee_mask,
                          r=0.015, min_cluster_size=10):
        needs_split = []
        for ei, entity_idx in enumerate(torch.unique(self.entity_indices)):
            if self.is_ripped[ei]:  # already ripped -> skip
                # print(f'entity {entity_idx} already ripped')
                needs_split += [False]
                continue
            if self.is_missed[ei]:  # already missed -> skip
                # print(f'entity {entity_idx} already missed')
                needs_split += [False]
                continue
            entity_mask = self.entity_indices == entity_idx
            check_mask = torch.logical_and(entity_mask, ~inee_mask)
            
            # check if top/bot are still connected
            check_graph = self._get_neighborhood_graph(particles_check[check_mask],
                                                       r=r)
            _, check_labels = self._get_particle_labels(check_graph, particles_check.shape[0],
                                                        min_cluster_size=min_cluster_size)
            # number of non-trivial components (i.e., not single particles)
            check_unique_labels, check_counts = torch.unique(check_labels, return_counts=True)
            check_components = (check_counts > 1).sum()

            needs_split += [bool(check_components > 1)]
        return np.any(needs_split), needs_split
    
    def get_between_region(self, particles, ee_pose, ee_width, ee_open):
        # transform all particles to grasp frame
        particles_in_ee_frame = torch.matmul(particles - ee_pose[:3, 3], ee_pose[:3, :3])
        # check against ee geometry -> inside hull == between fingers
        #   (bbox sufficient for square fingers)
        y_between_mask = torch.logical_and(-ee_width/2 <= particles_in_ee_frame[:, 1],
                                           particles_in_ee_frame[:, 1] <= ee_width/2)
        x_between_mask = torch.logical_and(-ee_open/2 <= particles_in_ee_frame[:, 0],
                                           particles_in_ee_frame[:, 0] <= ee_open/2)
        between_mask = torch.logical_and(x_between_mask, y_between_mask)        
        return between_mask.float()

    def _get_neighborhood_graph(self, particles, r=0.015, max_num_neighbors=32, mode='radius'):
        if mode == 'radius':
            edge_index = radius_graph(particles, r=r, loop=False, max_num_neighbors=max_num_neighbors)
        elif mode == 'knn':
            edge_index = knn_graph(particles, k=max_num_neighbors, loop=False)
        elif mode == 'approx_knn':
            edge_index = approx_knn_graph(particles, k=max_num_neighbors, loop=False)
        else:
            raise NotImplementedError(f'Unknown mode {mode}')
        return edge_index

    def _get_particle_labels(self, edge_index, num_nodes, min_cluster_size=5, ordered_by=''):
        # label based on connected components
        graph = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        n_components, cluster_labels = csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)
        # filter out small clusters
        unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        cluster_sizes[cluster_sizes < min_cluster_size] = 0
        n_components = np.sum(cluster_sizes > 0)  # update number of components
        # set labels of small clusters to -1 (outlier)
        cluster_labels = np.where(np.isin(cluster_labels, unique_labels[cluster_sizes > 0]), cluster_labels, -1)
        # reassign labels to get rid of gaps resulting from outliers, sort by size (descending)
        unique_labels, count_labels = np.unique(cluster_labels, return_counts=True)
        consecutive_label = 0
        old_cluster_labels = cluster_labels.copy()
        order = np.argsort(count_labels)[::-1] if ordered_by == 'size' else np.arange(unique_labels.shape[0])
        for old_label in unique_labels[order]:
            if old_label == -1:
                continue
            cluster_labels[old_cluster_labels == old_label] = consecutive_label
            consecutive_label += 1
        # back to torch
        cluster_labels = torch.from_numpy(cluster_labels).to(edge_index.device).float()
        return n_components, cluster_labels

    def _get_particle_labels_sliced(self, particles, particle_labels, edge_index, slice_thresholds, ordered_by=''):
        # label based on connected components, computed over slices and then merged
        per_slice_components = []
        for slice_lower, slice_upper in zip(slice_thresholds[0:-1], slice_thresholds[1:]):
            slice_mask = torch.logical_and(slice_lower < particles[:, 1], particles[:, 1] <= slice_upper)
            slice_subset = torch.argwhere(slice_mask).view(-1)
            if not np.all([idx in edge_index for idx in slice_subset]):
                per_slice_components.append(0)
                continue
            slice_graph = subgraph(slice_subset, edge_index, relabel_nodes=True)[0]
            slice_components, slice_labels = self._get_particle_labels(slice_graph, slice_subset.shape[0],
                                                                       ordered_by=ordered_by)
            per_slice_components.append(slice_components)
            particle_labels[slice_mask] = slice_labels + (2 if slice_components == 1 else 0)
        return particle_labels, np.array(per_slice_components)

    def _get_connected(self, check_particles, check_graph, ee_pose):
        # get particles in ee frame
        particles_in_ee_frame = torch.matmul(check_particles - ee_pose[:3, 3], ee_pose[:3, :3])

        slice_thresholds = self._get_slice_thresholds(particles_in_ee_frame)
        particle_labels = -torch.ones_like(particles_in_ee_frame[:, 0])
        particle_labels, per_slice_components = self._get_particle_labels_sliced(particles_in_ee_frame, particle_labels,
                                                                                 check_graph, slice_thresholds,
                                                                                 ordered_by='size')
        # self.show(check_particles, particle_labels)  # debug
        min_merged_slices = np.ceil(per_slice_components.shape[0] * 0.5)  # at least half
        connected = np.sum(per_slice_components <= 1) >= min_merged_slices
        return connected, particle_labels

    def _get_slice_thresholds(self, particles, slice_width=0.005):
        slice_limits = float(particles[:, 1].min()), float(particles[:, 1].max())
        # get full slices, centered around middle
        slice_range = slice_limits[1] - slice_limits[0]
        slice_mid = (slice_limits[1] + slice_limits[0]) / 2
        # check needs to be at least one slice wide for this to be > 0 -- else linspace is empty (only limits added)
        slice_count = int(np.floor((slice_mid - slice_limits[0]) / slice_width))
        slice_half_offset = slice_count * slice_width
        slice_thresholds = np.linspace(slice_mid - slice_half_offset, slice_mid + slice_half_offset, slice_count)
        if slice_half_offset*2 < slice_range:
            # add remainder slices (thinner than slice_width)
            slice_thresholds = np.concatenate([[slice_limits[0]], slice_thresholds, [slice_limits[1]]])
        # assert np.all([poi in slice_thresholds for poi in list(slice_limits) + [slice_mid]])
        return slice_thresholds

    def _label_to_color(self, labels):
        colors = torch.zeros((labels.shape[0], 3)).to(labels.device)
        def set_color(label, color):
            for i, v in enumerate(color):
                colors[labels == label, i] = v
        set_color(0, [1, 0, 1])  # magenta
        set_color(1, [0, 1, 1])  # cyan
        set_color(2, [0, 0, 1])  # blue
        set_color(-1, [0.7]*3)  # gray
        # anything else is black
        return colors

    def show(self, particles=None, labels=None):
        if particles is None:  # default: initial particles
            particles = self.particles_at_rest
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(particles.cpu()))
        pcd.paint_uniform_color([0.7]*3)
        if labels is None:  # default: components
            labels = self.entity_indices.cpu()
        colors = self._label_to_color(labels).cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


class Topology:

    """
    Our topology check involves adjusting particle velocities to move object regions 
    opposite to the end effector's closing direction (merge) or orthogonal to it (split).
    If the distance between these regions after perturbation by the checking action falls below (merge)
    or above (split) a threshold, this topological change is recorded in both 
    the particle-connectivity graph and the scene-topology graph.
    """

    def __init__(self, cfg, scene, planner, entities):
        self.cfg = cfg

        if 'donut' in self.cfg.entities.keys():
            scene_type = 'g1'  # c=1; e.g., donut
        elif 'left' in self.cfg.entities.keys():
            scene_type = 'g0g0'  # c=2; e.g., two rolls
        else:
            scene_type = 'g0'  # c=1; e.g., roll
        self.scene_type = scene_type

        # simulation
        self.scene = scene
        self.planner = planner
        self.entities = entities
        self.check_entities = [e for e in self.entities if e.name in self.cfg.check.entities]
        self.check_others = [e for e in self.entities if e.name not in self.cfg.check.entities]

        # topology data structures, created by reset
        self.scene_graph = None
        self.particle_graph = None

    def reset(self):
        # topology data structures
        self.scene_graph = SceneGraph(self.scene_type)
        self.particle_graph = ParticleGraph(self.scene_type, self.entities)
        color_particles(self.check_entities, self.particle_graph.entity_indices)

    def check(self):
        # save simulation state before checking (restored after)
        cur_state = self.scene.get_state()
        cur_ee_state = self.planner.ee.get_state()
        cur_ee_quat = cur_ee_state[3:7].clone()

        # get current state of particles and ee
        particles = get_scene_particles(self.check_entities)[0]
        ee_pose = torch.eye(4).to(particles.device)
        ee_pose[:3, :3] = quat_to_R(cur_ee_quat.clone())
        ee_pose[:3, 3] = cur_ee_state[:3].clone()
        ee_width = self.cfg.ee.entities['finger_left']['geom']['scale'][1]
        ee_open = float(cur_ee_state[7])

        # == 1) check if the ee missed any component - if all are missed, no further checks are needed
        if not self.is_merged() and not self.is_split():  # no further checks needed
            all_missed = self.particle_graph.check_missed(particles, ee_pose, ee_width)
            if self.particle_graph.check_ripped(particles):
                self.scene_graph.is_merged = True  # can no longer merge if the region of interest has been ripped away
        else:
            all_missed = False
        #
        need_merge_check = not self.is_merged() and not all_missed
        # assume that (self-)merge needs to happen first (i.e., single component left between ee) before split
        #   (plus, ee needs to be sufficiently closed - otherwise, we can skip the check)
        need_split_check = (self.is_merged() and not self.is_split()) and ee_open < 0.05 and not all_missed
        if not need_merge_check and not need_split_check:
            return  # no further checks needed; no state reset needed since we did not change anything

        # == 2) check if any component(s) (self-)merged
        if need_merge_check:
            # check if the two regions are touching already (-> requires sim-based check)
            merge_region, needs_sim = self.particle_graph.check_merge_before(particles, ee_pose, ee_width, ee_open)
            # self.particle_graph.show(particles, merge_region)  # debug: current particle state + regions to move apart

            if needs_sim:
                # prepare check: setting pos resets v, F, C
                n_start = 0
                for e in self.check_entities:
                    e.set_position(particles[n_start:n_start+e.n])
                    n_start += e.n

                # move the regions apart to check
                check_vel = torch.tensor(self.cfg.check.vel).to(us.FTYPE_TC).cuda()
                check_vel = transform_by_quat(check_vel, cur_ee_quat)
                # merge_region is 0 for left, 1 for right; ignore (-1 or >1) elsewhere
                merge_mask = torch.logical_or(merge_region == 0, merge_region == 1)
                # move_vels are negative for left, positive for right; zero elsewhere
                move_vels = (((merge_region - 0.5) * 2) * merge_mask)[..., None] * check_vel[None, ...]/2
                deactivate_ee(self.planner, self.check_others)  # to avoid collision with ee during check
                move_region(self.scene, self.check_entities, move_vels, self.cfg.check.horizon)

                # check regions after move -> if they are still touching, the regions merged
                merge_particles = get_scene_particles(self.check_entities)[0]
                # self.particle_graph.show(merge_particles, merge_mask)  # debug: particle state after move + potential merge region
                did_merge, inhull_labels = self.particle_graph.check_merge_after(merge_particles,
                                                                                 ee_pose, ee_width, ee_open,
                                                                                 merge_mask)
                # self.particle_graph.show(merge_particles, inhull_labels)  # debug: particle state after move + between-ee merge region
                
                if did_merge:
                    # collapse regions in particle and scene graph
                    for entity_idx in self.get_entity_indices():
                        # update particle graph
                        entity_mask = self.particle_graph.entity_indices == entity_idx
                        empty_regions, merge_regions = self.particle_graph.merge(particles, inhull_labels,
                                                                                 entity_mask, ee_pose, ee_width)
                        # update scene graph
                        for k, do_remove in zip(['above', 'below'], empty_regions):
                            if do_remove:
                                self.scene_graph.remove(k, entity_idx)
                        for k, do_merge in zip(['above', 'below'], merge_regions):
                            if do_merge:
                                self.scene_graph.merge(k, entity_idx)
                    # merge in scene graph
                    new_node_idx = self.scene_graph.merge('mid')  # merging mid in scene graph (merges self or components)
                    new_entity_idx = self.scene_graph.graph.nodes[new_node_idx]['comp']  # use label of new node
                    # merge in particle graph (i.e., update component labels)
                    # (in our case, merge is assumed to always happen first - reducing the scene to one component with all particles)
                    self.particle_graph.entity_indices = torch.ones(self.particle_graph.particles_at_rest.shape[0],
                                                                    dtype=torch.long, device=self.particle_graph.particles_at_rest.device) * new_entity_idx
                    self.particle_graph.entity_particle_counts = [sum(self.particle_graph.entity_particle_counts)]

                # reset state after check
                self.planner.ee.set_state(cur_ee_state)
                self.scene.set_state(cur_state)
        
        # == 3) check if any component(s) split
        if need_split_check:
            # check if the two regions above/below are non-empty (-> requires sim-based check)
            split_region, needs_check = self.particle_graph.check_split_before(particles, ee_pose, ee_width)
            # self.particle_graph.show(particles, split_region)  # debug: current particle state + regions to move apart

            if needs_check:
                # prepare check: setting pos resets v, F, C
                # (also, get rid of in-ee particles)
                padding = 0.005
                inhull_labels = self.particle_graph.get_between_region(particles, ee_pose, ee_width+padding, 0.5)
                between_labels = self.particle_graph.get_between_region(particles, ee_pose, ee_width+padding, ee_open-padding)
                inee_mask = torch.logical_and(inhull_labels == 1, between_labels == 0)
                #
                new_particles = particles.clone()
                new_particles[inee_mask] = 0.5  # place in some corner while checking
                n_start = 0
                for e in self.check_entities:
                    e.set_position(new_particles[n_start:n_start+e.n])
                    n_start += e.n

                # move apart
                check_vel = torch.tensor(self.cfg.check.vel).to(us.FTYPE_TC).cuda()
                check_vel = transform_by_quat(check_vel, cur_ee_quat)
                # check orthogonal to grasp
                quat_ortho = torch.tensor(euler_to_quat([0, 0, 90])).to(us.FTYPE_TC).cuda()
                check_vel = transform_by_quat(check_vel, quat_ortho)
                # split_region is 1 for above, 0 for below; ignore in-ee particles
                split_mask = (~inee_mask).float()
                # move_vels are negative for left, positive for right; zero elsewhere
                move_vels = (((split_region - 0.5) * 2) * split_mask)[..., None] * check_vel[None, ...]/2
                #
                deactivate_ee(self.planner, self.check_others)  # to avoid collision with ee during check
                move_region(self.scene, self.check_entities, move_vels, self.cfg.check.horizon)

                # check regions after move -> if they are still connected, the regions did not split
                split_particles = get_scene_particles(self.check_entities)[0]
                # self.particle_graph.show(split_particles, split_region)  # debug: particle state after move + potential split regions
                did_split, split_entities = self.particle_graph.check_split_after(split_particles, inee_mask)

                if did_split:
                    # split regions in particle and scene graph
                    for entity_idx, needs_split in zip(self.get_entity_indices(), split_entities):
                        if not needs_split:
                            continue
                        
                        # update scene graph: split into top and bottom parts at merge node
                        merge_nodes = [n for n in self.scene_graph.get_entity_nodes(entity_idx) if self.scene_graph.graph.nodes[n]['kind'] == 'merge']
                        new_above_idx, new_below_idx = self.scene_graph.split(merge_nodes[0])

                        # update particle graph
                        entity_mask = self.particle_graph.entity_indices == entity_idx
                        for i, new_node_idx in enumerate([new_below_idx, new_above_idx]):
                            # use label of new node in graph
                            new_entity_idx = self.scene_graph.graph.nodes[new_node_idx]['comp']
                            # fetch affected particles from original split regions (including in-ee region)
                            new_entity_mask = torch.logical_and(split_region == i, entity_mask)
                            # assign new entity index
                            self.particle_graph.entity_indices[new_entity_mask] = torch.ones(new_entity_mask.sum(), dtype=torch.long,
                                                                                             device=entity_mask.device) * new_entity_idx
                        self.entity_particle_counts = torch.unique(self.particle_graph.entity_indices, return_counts=True)[1].cpu().numpy().tolist()
        self.planner.ee.set_state(cur_ee_state)
        self.scene.set_state(cur_state)
        color_particles(self.check_entities, self.particle_graph.entity_indices)
        self.scene.step()

    def __str__(self) -> str:
        desc = f'{self.get_num_components()} component(s) of genus {self.get_genus_per_component()}'
        return desc

    def get_num_components(self):
        return self.scene_graph.get_num_components()
    
    def get_genus_per_component(self):
        return self.scene_graph.get_genus_per_component()
    
    def get_entity_indices(self):
        return torch.unique(self.particle_graph.entity_indices)

    def is_merged(self):
        if self.scene_type == 'g1':  # if True, means self merge
            return self.scene_graph.is_merged
        else:  # if True, means components merged -- or missed one component (in g0g0), so we're done too
            return self.scene_graph.is_merged or (~self.is_missed()).sum() < 2
    
    def is_split(self):
        return self.scene_graph.is_split
    
    def is_missed(self):
        return self.particle_graph.is_missed


def deactivate_ee(planner, check_others):
    # move ee away (changes pos/quat, zeros v/w - but keeps state machine and controller state)
    planner.ee.set_state(torch.tensor([1.0, 1, 1, 1, 0, 0, 0, 0]).to(us.FTYPE_TC).cuda())
    # remove velocities of other entities (any residual from interaction with ee); reverted later by applying cur_state
    for entity in check_others:
        entity.set_velocity(torch.tensor([0, 0, 0]).to(us.FTYPE_TC).cuda())

def move_region(scene, check_entities, move_vels, check_steps):
    for _ in range(check_steps):
        # set particle velocities
        n_start = 0
        for e in check_entities:
            e.set_velocity(move_vels[n_start:n_start+e.n])
            n_start += e.n
        # step sim
        scene.step()

def color_particles(check_entities, particle_labels):
    # labels to colors
    import matplotlib.cm as cm
    cmap = cm.get_cmap('Set2')
    colors = torch.tensor(cmap(particle_labels.cpu()).astype(np.float32), device=particle_labels.device)

    n_start = 0
    for e in check_entities:
        e.set_color(colors[n_start:n_start+e.n])
        n_start += e.n

def get_scene_particles(entities, what='pos', skip=['ground']):
    particles = []
    counts = []
    for entity in entities:
        if entity.name in skip:
            continue
        particles_val, count = get_entity_particles(entity, what=what)
        particles += [particles_val]
        counts += [count]
    return torch.cat(particles, dim=0), counts

def get_entity_particles(entity, what='pos'):
    assert isinstance(entity, us.engine.entities.mpm_entity.MPMEntity)

    num_particles = entity.n
    particles_state = entity.get_state()
    if what == 'pos':
        particles = particles_state.pos.detach()
    elif what == 'vel':
        particles = particles_state.vel.detach()
    return particles, num_particles

def get_entity_state(entity):
    assert isinstance(entity, us.engine.entities.mpm_entity.MPMEntity)

    num_particles = entity.n
    particles_state = entity.get_state()
    particles_pos = particles_state.pos.detach().cpu().numpy()
    particles_vel = particles_state.vel.detach().cpu().numpy()

    state = {
        'particles': {
            'num': num_particles,
            'pos': particles_pos,
            'vel': particles_vel,
        },
        'grid': {
            'resolution': entity.sim.mpm_solver.grid_res,
            'cell_size': entity.sim.mpm_solver.dx,
            'bounds': (entity.sim.mpm_solver.lower_bound.tolist(),
                       entity.sim.mpm_solver.upper_bound.tolist()),
            'bound_padding': entity.sim.mpm_solver.boundary_padding,
        },
    }
    return state
