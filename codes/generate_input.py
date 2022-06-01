from __future__ import print_function

import numpy as np
import random
import json
import networkx as nx

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]


def run_random_walks(G, nodes, walk_len=None, num_walks=None):
    """
    Runs an unbiased random walk on the given graph for all nodes
    Returns : array with pairs of co-occuring nodes
    """
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(walk_len):
                next_node = random.choice(G.neighbors(curr_node)) # Un-biased random walk
                # self co-occurrences are useless
                if curr_node != node:  # and (node, curr_node) not in pairs:
                    pairs.append((node, curr_node))

                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")

    print('Number of local neighbors:', len(pairs))

    return pairs


def run_dfs_walks(G, nodes, dfs_len=None, num_walks=None):
    """
     param G: Graph
     param id_map: graph nodes ids
     param nodes: graph nodes
     param dfs_len: Length of walks
     param num_walks: Number of nodes to sample for each given node
     return: dfs_pair of walks

        # number of walks from node,
        # each walk gives one node, curr_node
        # If all walks result in a node each node has a route with length of num_walks
       """
    dfs_pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            prev_node = None
            next_node = None
            for j in range(dfs_len):
                if prev_node is None:
                    depth_1_neighbors = [neigh for neigh in G.neighbors(curr_node)]
                    next_node = random.choice(depth_1_neighbors)
                depth_2_neighbors = [neigh for neigh in G.neighbors(next_node) if neigh not in G.neighbors(curr_node)]
                prev_node = curr_node
                curr_node = next_node
                next_node = random.choice(depth_2_neighbors)

        if curr_node != node:  # and (node, curr_node) not in dfs_pairs:
                dfs_pairs.append((node, curr_node))

        if count % 1000 == 0:
            print("Done DFS walks for", count, "nodes")

    print('Number of global neighbors:', len(dfs_pairs))

    return dfs_pairs


def convert_attributes_to_lists(g):
    for u, d in g.nodes(data=True):
        # print(d)
        for key, val in d.items():
            # print('here')
            # print(key, type(d[key]))
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    for u, v, d in g.edges(data=True):
        # print(d)
        for key, val in d.items():
            # print('here')
            # print(key, type(d[key]))
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    pass


def save_topological_pairs(G, path, PARAMS, bfs_walk=None, dfs_walk=None):

    WALK_LEN = PARAMS['walk_len']  # based sampling parameters
    WALK_NUM = PARAMS['walk_num']  # Number of nodes to sample for each given node
    prefix = '/' + PARAMS['prefix']

    # Extract training nodes/graphs
    nodes = [n for n in G.nodes() if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G = G.subgraph(nodes)

    # Run search mechanism
    if bfs_walk:
        pairs_bfs = run_random_walks(G, nodes, walk_len=WALK_LEN, num_walks=WALK_NUM)
        out_file = path + prefix + '-walks.txt'
        # Save into file
        with open(out_file, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs_bfs]))

    if dfs_walk:
        pairs_dfs = run_dfs_walks(G, nodes, dfs_len=2 * WALK_LEN, num_walks=WALK_NUM)
        out_file = path + prefix + "-dfs-walks.txt".format(WALK_NUM, 2 * WALK_LEN)
        # Save into file
        with open(out_file, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs_dfs]))


def save_data(L, path, prefix):

    prefix = '/' + prefix

    for n in L:
        for att in ['geometry', 'highway', 'maxspeed']:
            L.nodes[n].pop(att, None)

    convert_attributes_to_lists(L)

    # --------------- Save ID-Map
    id_map = {}
    for n in L.nodes:
        id_map[str(n)] = n

    out_file = path + prefix + '-id_map.json'
    with open(out_file, 'w') as fp:
        json.dump(id_map, fp)
    print('ID-Map saved in', out_file)

    # --------------- Save Class-Maps
    class_map = {}
    for n in L.nodes:
        class_map[str(n)] = np.array(L.nodes[n]['label']).astype(int).tolist()

    out_file = path + prefix + '-class_map.json'
    with open(out_file, 'w') as fp:
        json.dump(class_map, fp)
    print('Class-Map saved in', out_file)

    # --------------- Save Features
    data_arr = []
    out_file = path + prefix + '-feats.npy'
    for n, d in L.nodes(data=True):
        data_arr.append(np.hstack(
            ([
                d['midpoint'],
                np.array(d['maxspeed_one_hot']),
                np.array(d['geom']),
                d['length']
            ])))

    np.save(out_file, np.array(data_arr))
    print('Features saved in', out_file)

    # --------------- Save Graph
    for n in L:
        for att in ['length', 'label', 'geom', 'midpoint', 'maxspeed_one_hot']:
            L.nodes[n].pop(att, None)

    data = nx.json_graph.node_link_data(L)
    out_file = path + prefix + '-G.json'
    with open(out_file, 'w') as fp:
        json.dump(data, fp)
    print('Graph saved in', out_file)


