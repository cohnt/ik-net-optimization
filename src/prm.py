import numpy as np
from tqdm.auto import tqdm
import scipy
import time
import networkx as nx
from scipy.spatial import cKDTree as KDTree


class PRMOptions:
	def __init__(self, check_size=1e-2, k_neighbors=10, num_vertices=1e5, batch_size = 512, timeout=np.inf):
		self.check_size = check_size
		self.k_neighbors = k_neighbors
		self.num_vertices = num_vertices
		self.timeout = timeout
		self.batch_size = batch_size


class PRM:
	def __init__(self, RandomConfig, ValidityChecker, Distance=None):
		self.RandomConfig = RandomConfig
		self.ValidityChecker = ValidityChecker
		if Distance is None:
			self.Distance = lambda x, y: np.linalg.norm(x - y)
		else:
			self.Distance = Distance

		self.options = None
		self.tree = None

	def create_roadmap(self, start, goal, options):
		"""Create a PRM roadmap.

		Requirements (interfaces):
		- `RandomConfig(n)` MUST accept an integer `n` and return a numpy array of shape `(n, dim)`.
		- `ValidityChecker(configs)` MUST accept a numpy array of shape `(m, dim)` and return a numpy
		  boolean array of length `m` indicating validity for each configuration.

		Algorithm summary:
		- Draw `options.num_vertices` random configurations in batches using `RandomConfig` and keep the
		  valid ones as determined by `ValidityChecker` (batched).
		- Add `start` and `goal` as nodes 0 and 1.
		- Build a k-NN graph (k = options.k_neighbors) over all valid nodes.
		- For each candidate edge, discretize the straight-line in configuration space with steps of length
		  `options.check_size`, validate intermediate samples in batches edge-by-edge, stop checking an
		  edge as soon as a batch contains an invalid sample, and add edges whose intermediate samples are
		  all valid.
		"""

		t0 = time.time()
		self.options = options
		self.tree = nx.Graph()
		# Node 0: start, Node 1: goal
		start = np.asarray(start)
		goal = np.asarray(goal)
		self.tree.add_node(0, q=start)
		self.tree.add_node(1, q=goal)

		# number of random proposals to draw
		target_vertices = int(options.num_vertices)
		batch_size = int(options.batch_size)

		all_q_rows = [start, goal]
		n_drawn = 0
		pbar = tqdm(total=target_vertices, desc="Sampling PRM proposals")
		while n_drawn < target_vertices:
			if time.time() - t0 >= options.timeout:
				break
			n_to_draw = min(batch_size, target_vertices - n_drawn)
			batch = self.RandomConfig(n_to_draw)
			valid_batch = batch[self.ValidityChecker(batch)]
			if valid_batch.size > 0:
				all_q_rows.extend(valid_batch)
			n_drawn += n_to_draw
			pbar.update(n_to_draw)
		pbar.close()
		print(f"Found {len(all_q_rows) - 2} valid random configurations.")

		for i, q in enumerate(all_q_rows[2:], start=2):
			self.tree.add_node(i, q=np.asarray(q))

		# Build k-NN structure over all nodes
		all_q = np.vstack(all_q_rows)
		node_indices = list(self.tree.nodes)
		k = min(int(options.k_neighbors), max(len(node_indices) - 1, 1))
		tree = KDTree(all_q)
		_, idxs = tree.query(all_q, k=k + 1)

		print(f"Constructed k-NN graph with k={k} for {len(node_indices)} nodes. Preparing edge checks...")

		# Prepare candidate edges and their intermediate samples for validation
		edge_checks = []  # tuple (u_idx, v_idx, edge_samples)

		def _points_on_edge(q1, q2, step):
			dist = np.linalg.norm(q2 - q1)
			if dist <= step:
				return np.zeros((0, q1.shape[0]))
			n_steps = int(np.ceil(dist / step))
			# interior samples only
			ts = np.linspace(0.0, 1.0, n_steps + 1)[1:-1]
			pts = np.outer(1 - ts, q1) + np.outer(ts, q2)
			return pts

		average_points_tracker = []
		# Iterate over nodes and their neighbor indices to assemble edge sample requests
		for i_u, u in enumerate(node_indices):
			for j in range(1, idxs.shape[1]):
				v_pos = idxs[i_u, j]
				if v_pos >= len(node_indices):
					continue
				v = node_indices[v_pos]
				# add each undirected edge once (u < v by node id)
				if u >= v:
					continue
				q_u = self.tree.nodes[u]["q"]
				q_v = self.tree.nodes[v]["q"]
				pts = _points_on_edge(q_u, q_v, options.check_size)
				average_points_tracker.append(pts.shape[0])
				if pts.shape[0] == 0:
					# no intermediate checks required: add edge directly
					self.tree.add_edge(u, v, weight=np.linalg.norm(q_u - q_v))
					continue
				edge_checks.append((u, v, pts))

		print(f"Prepared {len(edge_checks)} edges for validation. Average intermediate points per edge: {np.mean(average_points_tracker):.2f}")
		
		if len(edge_checks) > 0:
			# Validate each edge in batches and stop checking as soon as a batch fails.
			pbar = tqdm(total=len(edge_checks), desc="Validating edges")
			for (u, v, pts) in edge_checks:
				edge_is_valid = True
				idx0 = 0
				while idx0 < pts.shape[0]:
					idx1 = min(idx0 + batch_size, pts.shape[0])
					sub = pts[idx0:idx1]
					valid_sub = self.ValidityChecker(sub)
					if not np.all(valid_sub):
						edge_is_valid = False
						break
					idx0 = idx1
				if edge_is_valid:
					q_u = self.tree.nodes[u]["q"]
					q_v = self.tree.nodes[v]["q"]
					self.tree.add_edge(u, v, weight=np.linalg.norm(q_u - q_v))
				pbar.update(1)
			pbar.close()

		return self.tree

	def find_path(self, tree=None, start=0, goal=1, return_configs=True):
		"""Find a path through a roadmap graph.

		Parameters:
		- tree: roadmap graph to search. Defaults to `self.tree`.
		- start, goal: node ids to connect.
		- return_configs: if True, return the sequence of configurations stored on each node.
		"""
		if tree is None:
			tree = self.tree

		if tree is None:
			return None

		try:
			node_path = nx.shortest_path(tree, source=start, target=goal, weight="weight")
		except (nx.NetworkXNoPath, nx.NodeNotFound):
			return None

		if not return_configs:
			return node_path

		return [tree.nodes[node]["q"] for node in node_path]

