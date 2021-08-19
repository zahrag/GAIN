from __future__ import print_function

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import random
import collections
from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import LineString
import json

import os
import sys


from GAIN.codes.generate_input import save_data, save_topological_pairs

'''
    Authors: 
    Zahra Gharaee (zahra.gharaee@liu.se)
    Oliver Stromann (oliver.stromann@liu.se)
    
    This script contains the codes to generate Inductive and Transductive datasets of road network graphs 
    extracted from OpenStreetMap (OSMnx). 
    '''


# ################# Extract Transductive dataset from OSM
def get_params_transductive():
    PARAMS = {
        # dataset parameters
        'prefix': 'linkoping-osm',
        'poi': (58.408909, 15.618521),
        'buffer': 7000,
        'geom_vector_len': 20,
        'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
                                    'access', 'lanes', 'oneway', 'name', 'key'],
        'exclude_node_attributes': ['ref', 'osmid'],

        # Original labels
        'label_lookup_': {'motorway': 0,
                          'trunk': 1,
                          'primary': 2,
                          'secondary': 3,
                          'tertiary': 4,
                          'unclassified': 5,
                          'residential': 6,
                          'motorway_link': 7,
                          'trunk_link': 8,
                          'primary_link': 9,
                          'secondary_link': 10,
                          'tertiary_link': 11,
                          'living_street': 12,
                          'road': 13,
                          'yes': 14,
                          'planned': 15
                          },
        # Merged labels
        'label_lookup': {'motorway': 0,
                         'trunk': 0,  # merge for class balance
                         'primary': 0,  # merge for class balance
                         'secondary': 0,  # merge for class balance
                         'tertiary': 4,
                         'unclassified': 5,
                         'residential': 6,
                         'motorway_link': 0,  # merge for class balance
                         'trunk_link': 0,  # merge for class balance
                         'primary_link': 0,  # merge for class balance
                         'secondary_link': 0,  # merge for class balance
                         'tertiary_link': 4,  # merge for class balance
                         'living_street': 12,
                         'road': 13,
                         'yes': 0,
                         'planned': 13
                         },

        # sampling parameters
        'sampling_seed': 42,
        'n_test': 1000,
        'n_val': 500,

        # random walk parameters
        'walk_seed': 42,
        'walk_len': 5,
        'walk_num': 50
    }

    return PARAMS


def sample_nodes(node_list, n_samples):
    samples = []
    np.random.shuffle(node_list)
    while len(node_list) != 0 and len(samples) < n_samples:
        samples.append(node_list.pop())

    return node_list, samples


def split_train_test_val_nodes(g, PARAMS):
    np.random.seed(PARAMS['sampling_seed'])
    remain_nodes, test_nodes = sample_nodes(list(g.nodes), PARAMS['n_test'])
    _, val_nodes = sample_nodes(remain_nodes, PARAMS['n_val'])

    test_dict = {}
    val_dict = {}

    for n in g.nodes:  # default
        test_dict[n] = False
        val_dict[n] = False
    for n in test_nodes:
        test_dict[n] = True
    for n in val_nodes:
        val_dict[n] = True

    nx.set_node_attributes(g, test_dict, 'test')
    nx.set_node_attributes(g, val_dict, 'val')
    pass


def extract_osm_network_transductive():

    PARAMS = get_params_transductive()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")
    # neptune.log_text('timestamp', timestamp)

    # Retrieve osm data by center coordinate and spatial buffer
    g = ox.graph_from_point(PARAMS['poi'], dist=PARAMS['buffer'], network_type='drive', simplify=True)
    g = ox.project_graph(g, to_crs="EPSG:32633")

    g.graph['osm_query_date'] = timestamp
    g.graph['name'] = PARAMS['prefix']
    g.graph['poi'] = PARAMS['poi']
    g.graph['buffer'] = PARAMS['buffer']

    # create incremental node ids
    g = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default')

    # convert to undirected graph (i.e. directions and parallel edges are removed)
    g = nx.Graph(g.to_undirected())

    return g, PARAMS


# ################# Extract Inductive dataset from OSM
def get_PARAMS_inductive():

    # Biggest Swedish cities
    # Source: https://population.mongabay.com/population/sweden/
    PLACES = {
        # 'Stockholm': (0, 0), # >1,500,000 population
        # 'Göteborg': (0, 0), # > 500,000
        # 'Malmö': (0, 0), # > 300,000
        'Uppsala': (59.857994, 17.638622),  # < 150,000
        # 'Sollentuna': (0, 0), #OBS Stockholm förort
        # 'Södermalm': (0, 0), #OBS Stockholm förort
        'Västerås': (56.609789, 16.544657),
        'Örebro': (59.274752, 15.214113),
        'Linköping': (58.408909, 15.618521),
        'Helsingborg': (56.046472, 12.695231),
        'Jönköping': (57.782611, 14.162930),
        'Norrköping': (58.586859, 16.193182),
        # 'Huddinge': (0, 0), #OBS Stockholm förort
        'Lund': (55.703863, 13.191811),
        'Umeå': (63.825855, 20.265303),
        # 'Haninge': (0, 0), #OBS Stockholm förort
        'Gävle': (60.674963, 17.141546),
        'Borås': (57.721223, 12.939515),
        'Södertälje': (59.194800, 17.626693),
        # 'Kungsholmen': (0, 0), #OBS Stockholm förort
        'Eskilstuna': (59.370546, 16.509992),
        # 'Solna': (0, 0),#OBS Stockholm förort
        'Halmstad': (56.673874, 12.863075),
        'Växjö': (56.877798, 14.907140),
        'Karlstad': (59.403223, 13.512568),
        # 'Bromma': (0, 0),#OBS Stockholm förort
        # 'Mölndal': (0, 0), #OBS Göteborg förort
        # 'Vasastan': (0, 0),
        # 'Täby': (0, 0),
        'Sundsvall': (62.392445, 17.305561)  # > 50,000

    }

    PARAMS = {
        # dataset parameters
        'prefix': 'sweden-osm',
        'places': PLACES,
        'buffer': 7000,
        'geom_vector_len': 20,
        'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
                                    'access', 'lanes', 'oneway', 'name', 'key'],
        'exclude_node_attributes': ['ref', 'osmid'],

        # Original labels
        'label_lookup_': {'motorway': 0,
                          'trunk': 1,
                          'primary': 2,
                          'secondary': 3,
                          'tertiary': 4,
                          'unclassified': 5,
                          'residential': 6,
                          'motorway_link': 7,
                          'trunk_link': 8,
                          'primary_link': 9,
                          'secondary_link': 10,
                          'tertiary_link': 11,
                          'living_street': 12,
                          'road': 13,
                          'yes': 14,
                          'planned': 15
                          },

        # Merged labels
        'label_lookup': {'motorway': 0,
                         'trunk': 0,  # merge for class balance
                         'primary': 0,  # merge for class balance
                         'secondary': 0,  # merge for class balance
                         'tertiary': 4,
                         'unclassified': 5,
                         'residential': 6,
                         'motorway_link': 0,  # merge for class balance
                         'trunk_link': 0,  # merge for class balance
                         'primary_link': 0,  # merge for class balance
                         'secondary_link': 0,  # merge for class balance
                         'tertiary_link': 4,  # merge for class balance
                         'living_street': 12,
                         'road': 5,
                         'yes': 0,
                         'planned': 5
                         },


        # sampling parameters
        'sampling_seed': 1337,
        'n_test': 2,  # whole graphs!
        'n_val': 2,  # whole graphs!bla

        # random walk parameters
        'walk_seed': 42,
        'walk_len': 5,
        'walk_num': 50
    }

    return PARAMS


def split_train_test_val_graphs(PARAMS):  # for whole graphs in places

    places = PARAMS['places']

    print('total set size:', len(places))
    random.seed(PARAMS['sampling_seed'])
    test_places = random.sample(list(places.items()), PARAMS['n_test'])
    for place in test_places:
        del places[place[0]]
    val_places = random.sample(list(places.items()), PARAMS['n_val'])
    for place in val_places:
        del places[place[0]]
    print('training set size:', len(places))
    print('training set size:', len(test_places))
    print('training set size:', len(val_places))
    return places, dict(test_places), dict(val_places)


def extract_osm_network_inductive():

    PARAMS = get_PARAMS_inductive()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")

    places, test_places, val_places = split_train_test_val_graphs(PARAMS)

    sub_Gs = []
    print('Training set')
    for poi in places:
        print('Extracting road network for', poi, places[poi])
        sub_G = nx.Graph(ox.graph_from_point(places[poi], dist=PARAMS['buffer'], network_type='drive',
                                             simplify=True).to_undirected())
        print(nx.info(sub_G), '\n')
        nx.set_node_attributes(sub_G, False, 'test')
        nx.set_node_attributes(sub_G, False, 'val')
        nx.set_edge_attributes(sub_G, False, 'test')
        nx.set_edge_attributes(sub_G, False, 'val')
        sub_Gs.append(sub_G)

    print('Test set')
    for poi in test_places:
        print('Extracting road network for', poi, test_places[poi])
        sub_G = nx.Graph(ox.graph_from_point(test_places[poi], dist=PARAMS['buffer'], network_type='drive',
                                             simplify=True).to_undirected())
        print(nx.info(sub_G), '\n')
        nx.set_node_attributes(sub_G, True, 'test')
        nx.set_node_attributes(sub_G, False, 'val')
        nx.set_edge_attributes(sub_G, True, 'test')
        nx.set_edge_attributes(sub_G, False, 'val')
        sub_Gs.append(sub_G)

    print('Validation set')
    for poi in val_places:
        print('Extracting road network for', poi, val_places[poi])
        sub_G = nx.Graph(ox.graph_from_point(val_places[poi], dist=PARAMS['buffer'], network_type='drive',
                                             simplify=True).to_undirected())
        print(nx.info(sub_G), '\n')
        nx.set_node_attributes(sub_G, False, 'test')
        nx.set_node_attributes(sub_G, True, 'val')
        nx.set_edge_attributes(sub_G, False, 'test')
        nx.set_edge_attributes(sub_G, True, 'val')
        sub_Gs.append(sub_G)

    g = nx.compose_all(sub_Gs)
    g = ox.project_graph(g, to_crs="EPSG:32633")
    # graph attributes
    g.graph['osm_query_date'] = timestamp
    g.graph['name'] = PARAMS['prefix']
    # create incremental node ids
    g = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default')
    # convert to undirected graph (i.e. directions and parallel edges are removed)
    g = nx.Graph(g.to_undirected())

    return g, PARAMS


# ################# Generate graphs
def convert_class_labels(g, PARAMS):

    cnt = 0
    labels = nx.get_edge_attributes(g, 'highway')
    labels_int = {}
    for edge in g.edges:
        # set default attributes
        if not edge in labels:
            labels[edge] = 'road'

        # some edges have two attributes, take only their first
        if type(labels[edge]) == list:
            labels[edge] = labels[edge][0]

        # some edges have attributes, not listed in our label lookup
        # these could be added to the label lookup if increases significantly
        if not labels[edge] in PARAMS['label_lookup']:
            cnt += 1
            labels[edge] = 'road'

        print('Number of newly added road labels by OSM:', cnt)
        labels_int[edge] = PARAMS['label_lookup'][labels[edge]]

    nx.set_edge_attributes(g, labels_int, 'label')
    pass


def remove_unwanted_attributes(g, PARAMS):

    # deleting some node attributes
    for n in g:
        for att in PARAMS['exclude_node_attributes']:
            g.nodes[n].pop(att, None)
    # deleting some edge attributes
    for n1, n2, d in g.edges(data=True):
        for att in PARAMS['exclude_edge_attributes']:
            d.pop(att, None)
    pass


def standardize_geometries(g, PARAMS, attr_name='geom', verbose=0):

    steps = PARAMS['geom_vector_len']

    if verbose > 0:
        print('\nGenerating fixed length (%d) geometry vectors...' % (steps))
    geoms = nx.get_edge_attributes(g, 'geometry')
    xs = nx.get_node_attributes(g, 'x')
    ys = nx.get_node_attributes(g, 'y')
    np_same_length_geoms = {}
    count_no = 0
    count_yes = 0
    for e in g.edges():
        points = []

        if e not in geoms:  # edges that don't have a geometry
            line = LineString([(xs[e[0]], ys[e[0]]), (xs[e[1]], ys[e[1]])])
            for step in np.linspace(0, 1, steps):
                point = line.interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_no += 1

        else:  # all other edges
            for step in np.linspace(0, 1, steps):
                point = geoms[e].interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_yes += 1
        np_same_length_geoms[e] = np.array([np.array((p[0], p[1])) for p in points])

    if verbose > 0:
        print('- Geometry inserted from intersection coordinates for', count_no, 'nodes.')
        print('- Standardized geometry created for', count_no + count_yes, 'nodes.')

    nx.set_edge_attributes(g, np_same_length_geoms, attr_name)
    if verbose > 0:
        print('Done.')
    pass


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def midpoint_generation(g):
    pos = {}
    for u, d in g.nodes(data=True):
        pos[u] = (d['x'], d['y'])
    new_pos = {}
    for u, v, d in g.edges(data=True):
        e = (u, v)
        new_pos[e] = {'midpoint': np.array(midpoint(pos[u], pos[v]))}
    nx.set_edge_attributes(g, new_pos)
    pass


def midpoint_subtraction(g):
    for u, v, d in g.edges(data=True):
        e = (u, v)
        d['geom'] = d['geom'] - d['midpoint']
    pass


def one_hot_encode_maxspeeds(g, verbose=0):
    if verbose > 0:
        print('\nGenerating one-hot encoding maxspeed limits...')

    maxspeeds_standard = ['5', '7', '10', '20', '30', '40', '50', '60',
                          '70', '80', '90', '100', '110', '120', 'unknown']

    maxspeeds = nx.get_edge_attributes(g, 'maxspeed')
    maxspeeds_single_val = {}
    for e in g.edges():
        if e not in maxspeeds:
            maxspeeds[e] = 'unknown'

        if type(maxspeeds[e]) == list:
            maxspeeds_single_val[e] = maxspeeds[e][0]
        else:
            maxspeeds_single_val[e] = maxspeeds[e]

    for e in maxspeeds_single_val:
        if maxspeeds_single_val[e] not in maxspeeds_standard:
            if maxspeeds_single_val[e].isdigit():
                maxspeeds_standard.append(maxspeeds_single_val[e])
            else:
                maxspeeds_single_val[e] = 'unknown'

    enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(np.array(list(maxspeeds_single_val.values())).reshape(-1, 1))

    enc.fit(np.array(maxspeeds_standard).reshape(-1, 1))

    if verbose > 0:
        print('- One-hot encoder fitted to data with following categories:')
        print('-', np.array(enc.categories_).flatten().tolist())

    maxspeeds_one_hot = {k: enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist() for k, v in
                         maxspeeds_single_val.items()}
    if verbose > 0:
        print('- One-hot encoded maxspeed limits generated.')

    nx.set_edge_attributes(g, maxspeeds_one_hot, 'maxspeed_one_hot')
    if verbose > 0:
        print('Done.')
    pass


def generate_graph(g, PARAMS,
                   convert_labels=True, remove_unwanted=True, one_hot_maxspeed=True, standardize_geoms=True, verbose=0):

    if convert_labels:
        convert_class_labels(g, PARAMS)

    if remove_unwanted:
        remove_unwanted_attributes(g, PARAMS)

    if standardize_geoms:
        standardize_geometries(g, PARAMS, verbose=verbose)
    midpoint_generation(g)
    midpoint_subtraction(g)

    if one_hot_maxspeed:
        one_hot_encode_maxspeeds(g, verbose=verbose)

    return g


# ################# Print Graph Distributions
def count_frequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    od = collections.OrderedDict(sorted(freq.items()))

    for key, value in od.items():
        print("% d : % d" % (key, value))


def print_class_distribution(g, setting=None):

    if setting is 'osm_transductive':
        g_number_of_nodes = g.number_of_nodes()
        g_number_of_edges = g.number_of_edges()
        print('Train')
        print('number of Train nodes', g_number_of_nodes)
        print('number of Train edges', g_number_of_edges)

    else:
        print('Train')
        g_sub = g.edge_subgraph([(u, v) for u, v, d in g.edges(data=True) if not d['val'] and not d['test']])
        labels = nx.get_edge_attributes(g_sub, 'label')
        count_frequency((list(labels.values())))
        print('Test')
        g_sub = g.edge_subgraph([(u, v) for u, v, d in g.edges(data=True) if d['test']])
        labels = nx.get_edge_attributes(g_sub, 'label')
        count_frequency((list(labels.values())))
        print('Val')
        g_sub = g.edge_subgraph([(u, v) for u, v, d in g.edges(data=True) if d['val']])
        labels = nx.get_edge_attributes(g_sub, 'label')
        count_frequency((list(labels.values())))


# ################# Create line graph (L) transformed applying line-graph-transformation on original graph (G)
def copy_edge_attributes_to_nodes(g, l, verbose=0):
    if verbose > 0:
        print('Copying old edge attributes new node attributes...')
    node_attr = {}
    for u, v, d in g.edges(data=True):
        node_attr[(u, v)] = d
    nx.set_node_attributes(l, node_attr)


def convert_to_line_graph(g, setting, copy_attributes=True, verbose=1):
    # print input graph summary
    if verbose > 0:
        print('\n---Original Graph---')
        print(nx.info(g))

    # make edges to nodes, create edges where common nodes existed
    if verbose > 0:
        print('\nConverting to line graph...')
    l = nx.line_graph(g)

    # copy graph attributes
    l.graph['name'] = g.graph['name'] + '_line'
    l.graph['osm_query_date'] = g.graph['osm_query_date']
    l.graph['name'] = g.graph['name']

    # copy edge attributes to new nodes
    if copy_attributes:
        copy_edge_attributes_to_nodes(g, l, verbose=verbose)

    # relabel new nodes, storing old id in attribute
    mapping = {}
    for n in l:
        mapping[n] = n
    nx.set_node_attributes(l, mapping, 'original_id')
    l = nx.relabel.convert_node_labels_to_integers(l, first_label=0, ordering='default')

    # print output graph summary
    if verbose > 0:
        print('\n---Converted Graph---')
        print(nx.info(l))
        print('Done.')

    return l


# ################# Draw original and line graphs G & L
def get_pos(g):
    x = nx.get_node_attributes(g, 'x')
    y = nx.get_node_attributes(g, 'y')

    pos = {}
    for n in g:
        pos[n] = (x[n], y[n])

    return pos


def get_midpoint(g):
    pos = {}
    for u, d in g.nodes(data=True):
        # print(u)
        pos[u] = d['midpoint']

    return pos


def draw_graph(G, L):

    # show both graphs plus class labels
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)

    # plot original graph
    nx.draw_networkx(G, pos=get_pos(G), ax=ax, with_labels=False,
                     node_size=25, node_color='#999999', edge_color='#999999', width=3, alpha=0.7)

    # plot line graph
    nx.draw_networkx(L, pos=get_midpoint(L), ax=ax, with_labels=False,
                     node_size=25, node_color='black', edge_color='darkred', width=3, alpha=0.7)

    # associate labels with names and colors
    label_values = [0, 4, 5, 6, 12, 13]
    label_names = ['major', 'tertiary', 'unclassified', 'residential', 'living street', 'road']
    colors = ['red', 'orange', 'yellow', 'skyblue', 'lime', 'purple']

    # plot line graph node labels
    for c,label, label_name in zip(colors, label_values, label_names):
        L_sub = L.subgraph([n for n,v in L.nodes(data=True) if v['label'] == label])
        nx.draw_networkx(L_sub, pos=get_midpoint(L_sub), ax=ax, with_labels=False,
                         node_size=15, node_color=c, edge_color=c, width=0, label=label_name)

    plt.show()


if __name__ == "__main__":

    # settings
    Settings = ['osm_inductive', 'osm_transductive']
    setting = Settings[1]

    drawing_graphs = False
    saving_graphs_data = True
    saving_topological_neighborhood = False
    load_graphs = False

    # path
    mainpath = '../graph_data/'
    path = os.path.join(mainpath, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if setting == 'osm_inductive':
        G, PARAMS = extract_osm_network_inductive()
        G = generate_graph(G, PARAMS, verbose=1)
        L = convert_to_line_graph(G, setting)

    else:
        G, PARAMS = extract_osm_network_transductive()
        G = generate_graph(G, PARAMS, verbose=1)
        L = convert_to_line_graph(G, setting)
        split_train_test_val_nodes(L, PARAMS)

    if saving_graphs_data:
        save_data(L, path, PARAMS['prefix'])

    if saving_topological_neighborhood:
        save_topological_pairs(L, path, PARAMS, bfs_walk=True, dfs_walk=True)

    if drawing_graphs:
        draw_graph(G, L)

    if load_graphs:
        L = json.load(open(path + '/' + PARAMS['prefix'] + "-G.json"))


