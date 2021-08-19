from __future__ import division
from __future__ import print_function

from codes.layers import Layer

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    adj_info : Matrix of shape (V+1, max_node_degree)
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        """
        inputs : (ids, num_samples)
        ids : the node ids whose neighbors are to be sampled
        num_samples : number of samples to keep for each id

        This looks at the neighbor information of node ids
        It randomly shuffles the neighbor indices from adj_list and returns the first num_samples elements
        This is done for all nodes in the matrix

        Should return an array of shape (len(ids), num_samples) (len(ids) = batch_size generally speaking)
        """
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids) 
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        return adj_lists
