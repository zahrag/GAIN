import numpy as np
import json
import os

from networkx.readwrite import json_graph


def load_data(prefix, dir_data, normalize=True, walk_type=None, dfs_num_len=None):

    """
        Returns :
        G : Networkx graph. G.nodes[some_node_structure] = [features...]
        feats : Features of all nodes with scaled for train nodes
        id_map : A json-stored dictionary mapping the graph node ids to consecutive integers
        walks : random walk co-occurrences (format same as output of random_walk generation function given below)
        class_map : A json-stored dictionary mapping the graph node ids to classes

        """

    # ------ Load graph
    G_data = json.load(open(dir_data + prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    # ------ Load features/attributes
    if os.path.exists(dir_data + prefix + "-feats.npy"):
        feats = np.load(dir_data + prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None

    # ------ Load id-map
    id_map = json.load(open(dir_data + prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}

    # ------ Load class-map
    class_map = json.load(open(dir_data + prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)
    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    # ------ Remove all nodes that do not have val/test annotations
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print(
        "Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    # ------ Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now pre-processing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    # ------ Load walk-pairs
    rand_walks_file = "-walks.txt"
    dfs_walks_file = "-dfs-walks.txt".format(dfs_num_len[0], dfs_num_len[1])
    walks = []  # rand_edges

    if walk_type is 'rand_bfs_walks':
        with open(dir_data + prefix + rand_walks_file) as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    elif walk_type is 'rand_bfs_dfs_walks':
        with open(dir_data + prefix + rand_walks_file) as fp1:
            for line in fp1:
                walks.append(map(conversion, line.split()))

        with open(dir_data + prefix + dfs_walks_file) as fp2:
            for line in fp2:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

