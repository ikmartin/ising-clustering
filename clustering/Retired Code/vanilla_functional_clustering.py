class HierarchicalClustering_Vanilla:
    def __init__(self, distance_metric: Callable, refine_criterion: Callable, refine_method: Callable):
        self.distance_metric = distance_metric
        self.refine_criterion = refine_criterion
        self.refine_method = refine_method

    def _build_tree(self, node: Node):
        current_node_indices = self._cluster_map[node.value]
        if self.refine_criterion(self._data, self.distance_metric, current_node_indices):
            # Obtain split cluster tuples
            cluster1, cluster2 = self.refine_method(self._data, self.distance_metric, current_node_indices)

            # Register the hashes of the new clusters in the lookup table
            cluster1_key, cluster2_key = hash(cluster1), hash(cluster2)
            self._cluster_map.update({cluster1_key: cluster1, cluster2_key: cluster2})

            # Build child trees and add the child trees of the new clusters to the tree
            node.left, node.right = self._build_tree(Node(cluster1_key)), self._build_tree(Node(cluster2_key))

        return node

    def __call__(self, data: Spinspace):
        # Clusters will be kept track of as tuples of indices of spin states. The initial cluster is therefore
        # simply a tuple of all indices of spin states in the spin space.
        all_nodes_list = tuple(range(data.size))
        all_nodes_key = hash(all_nodes_list)

        # Binary trees from the binarytree package have nodes that can only store values of int, str, or float.
        # As such, we will keep a lookup table to associate integer hashes to the lists that they refer to.
        # Variables for the recursion process will be stored as private local variables to reduce the amount of
        # arguments that have to be passed around in the recursion.
        self._cluster_map = {all_nodes_key: all_nodes_list}
        self._data = data
        return self._build_tree(Node(all_nodes_key)), self._cluster_map