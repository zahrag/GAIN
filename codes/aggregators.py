import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros


class GAINAggregator(Layer):
    """
        Graph Attention Isomorphism Network "GAIN"
        Authors:
        Zahra Gharaee (zahra.gharaee@liu.se)
        Shreyas Kowshik (shreyaskowshik@iitkgp.ac.in)

        """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 transformed_dim=None, dropout=0., bias=False, num_multi_head=1, act=tf.nn.relu,
                 name=None, concat=False, model_size="small", **kwargs):
        super(GAINAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.num_multi_head = num_multi_head

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = 128
        elif model_size == "big":
            hidden_dim = 256
        self.hidden_dim = hidden_dim

        if transformed_dim is None:
            transformed_dim = output_dim
        self.transformed_dim = transformed_dim

        '''
            False: Identity function.
            True: Non-linearity ELU function.
            '''
        self.non_linearity = False

        '''
            False: One head and transformed weights used.
            True: One or more heads with different weights for self and neighbors.
            Both cases are tested using our settings, and the results are presented in the GAIN paper and Appendix.
            '''
        self.multi_head = False

        '''
            The role of epsilon could be studied further, however an exhaustive grid search is conducted over the epsilon
            in three different cases:
            zero epsilon, learning epsilon initialized with 0.001, 0.5.
            '''
        self.learning_epsilon = True
        if self.learning_epsilon:
            self.epsilon = tf.Variable(0.5)
        else:
            self.epsilon = 0

        # Create MLP
        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=transformed_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.leaky_relu,
                                     dropout=dropout,
                                     BN=True,
                                     sparse_inputs=False,
                                     logging=self.logging))

        for _ in range(1):
            self.mlp_layers.append(Dense(input_dim=hidden_dim,
                                         output_dim=hidden_dim,
                                         act=tf.nn.leaky_relu,
                                         dropout=dropout,
                                         BN=True,
                                         sparse_inputs=False,
                                         logging=self.logging))

        self.mlp_layers.append(Dense(input_dim=hidden_dim,
                                     output_dim=output_dim,
                                     act=tf.nn.leaky_relu,
                                     dropout=dropout,
                                     BN=True,
                                     sparse_inputs=False,
                                     logging=self.logging))

        # Attention Related Weights
        for i_ in range(self.num_multi_head):
            with tf.variable_scope(self.name + name + '_vars'):
                # Transformation to new representation: higher level features
                self.vars['transform_weights' + str(i_)] = glorot([input_dim, transformed_dim],
                                                                  name='transform_weights' + str(i_))
                # Feedforward network weight
                # 2 to account for concatenation
                self.vars['attention_weights' + str(i_)] = glorot([2 * transformed_dim, 1],
                                                                  name='attention_weights' + str(i_))

        # Multi-head attention
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([transformed_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):

        """
            This function aggregates vectors of the neighbors using GAIN.
            It uses different weights for the aggregated vectors of neighbors and the self vector if multi-head GAIN
            is applied, otherwise one-head using transformed weights is applied.
            The obtained representations followed by multiset function: MLP

            """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # self-attention (see explanation before eq.3 of GAT!)
        self_attention = False
        if self_attention:
            self_vecs_reshaped = tf.expand_dims(self_vecs, axis=1)  # Expand dimensions
            neigh_vecs = tf.concat([self_vecs_reshaped, neigh_vecs], axis=1)

        # Stores aggregated vectors for all heads in attention
        neigh_aggregated_heads = []
        for i_ in range(self.num_multi_head):

            '''
            Attention Layer
            Transform node and neighboring vectors
            Concatenate and pass through feedforward layer
            Compute attention weights and get output
            Concatenate self-vectors to neigh-vectors to include self-attention
            '''
            nv_shape = neigh_vecs.get_shape().as_list()

            neigh_transformed = tf.matmul(tf.reshape(neigh_vecs, [-1, nv_shape[-1]]),
                                          self.vars['transform_weights' + str(i_)])
            neigh_transformed = tf.reshape(neigh_transformed, [-1, nv_shape[1], self.transformed_dim])

            self_transformed = tf.matmul(self_vecs, self.vars['transform_weights' + str(i_)])

            # Neural Network
            self_transformed_tiled = tf.tile(tf.expand_dims(self_transformed, axis=1), [1, nv_shape[1], 1])
            concat_transformed = tf.concat([self_transformed_tiled, neigh_transformed], axis=-1)
            # Feedforward
            concat_shape = concat_transformed.get_shape().as_list()
            attention_out = tf.matmul(tf.reshape(concat_transformed, [-1, 2 * self.transformed_dim]),
                                      self.vars['attention_weights' + str(i_)])
            attention_out = tf.nn.leaky_relu(attention_out)
            attention_out = tf.reshape(attention_out, [-1, concat_shape[1]])

            # Apply soft-max row-wise
            attention_weights = tf.nn.softmax(attention_out, axis=1)
            attention_weights = tf.reshape(attention_weights, [-1, concat_shape[1], 1])

            # Get final hidden vectors
            neigh_weighted = neigh_transformed * attention_weights
            neigh_aggregated = tf.reduce_sum(neigh_weighted, axis=1)  # Sum column wise

            neigh_aggregated_heads.append(neigh_aggregated)

        concatenated_heads = tf.concat([tf.expand_dims(t, axis=1) for t in neigh_aggregated_heads], axis=1)
        # replace mean with sum operation
        final_neigh_aggregated = tf.reduce_sum(concatenated_heads, axis=1)

        # Apply Non-linearity: ELU or Identity function
        if self.non_linearity:
            final_neigh_aggregated = tf.nn.elu(final_neigh_aggregated)

        if self.multi_head:
            # One or more heads with different weights for self and neighbors
            from_neighs = tf.matmul(final_neigh_aggregated, self.vars['neigh_weights'])
            from_self = tf.matmul(self_vecs, self.vars['self_weights'])
            output = tf.add_n([(1 + self.epsilon) * from_self, from_neighs])

        else:
            # One head and transformed weights used
            output = tf.add_n([(1 + self.epsilon) * self_transformed, final_neigh_aggregated])

        # Apply multiset function: MLP
        for layer in self.mlp_layers:
            output = layer(output)

        return output


class AttentionAggregator(Layer):
    """
        Graph Attention Network "GAT" (https://arxiv.org/abs/1710.10903)
        Authors:
        Zahra Gharaee (zahra.gharaee@liu.se)
        Shreyas Kowshik (shreyaskowshik@iitkgp.ac.in)

        """

    def __init__(self, input_dim, output_dim, transformed_dim=None, num_multi_head=1, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        """
        input_dim : size of input features
        output_dim : size of output features
        transformed_dim : size of dimension for mapping vectors before attention calculation (W => input_dim -> transformed_dim)
        num_multi_head : number of multi-head-attention units. Helps in regularization
        """
        super(AttentionAggregator, self).__init__(**kwargs)

        self.num_multi_head = num_multi_head
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if transformed_dim is None:
            transformed_dim = input_dim

        self.transformed_dim = transformed_dim
        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([transformed_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([transformed_dim, output_dim],
                                               name='self_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        for i_ in range(self.num_multi_head):
            with tf.variable_scope(self.name + name + '_vars'):
                # Transformation to new representation: higher level features
                self.vars['transform_weights' + str(i_)] = glorot([input_dim, transformed_dim],
                                                                  name='transform_weights' + str(i_))
                # Feedforward network weight
                self.vars['attention_weights' + str(i_)] = glorot([2 * transformed_dim, 1],
                                                                  name='attention_weights' + str(i_))

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):

        """
            This function aggregates vectors of the neighbors using GAT.
            It transforms self and neighbor representation vectors to a new representation to get higher level features.
            It aggregated self and neighbor features using attention weights.
            It then uses different weights for the aggregated vectors and the self vector.
            It concatenates the obtained representations followed by non-linearity.
            inputs : self_vecs, neigh_vecs
            self_vecs : 2D tensor of shape [num_nodes_whose_vectors_needed, feature_size]
            neigh_vecs : 3D tensor of shape [num_nodes_whose_vectors_needed, num_samples_for_each_node, feature_size]
            """
        self_vecs, neigh_vecs_ = inputs
        neigh_vecs = tf.nn.dropout(neigh_vecs_, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # self-attention (study it further according to explanation before eq.3 of GAT!)
        self_attention = False
        if self_attention:
            self_vecs_reshaped = tf.expand_dims(self_vecs, axis=1)  # Expand dimensions
            neigh_vecs = tf.concat([self_vecs_reshaped, neigh_vecs], axis=1)

        # Stores aggreagted vectors for all heads in attention
        neigh_aggregated_heads = []

        neigh_weighted = None
        for i_ in range(self.num_multi_head):
            nv_shape = neigh_vecs.get_shape().as_list()

            neigh_transformed = tf.matmul(tf.reshape(neigh_vecs, [-1, nv_shape[-1]]),
                                          self.vars['transform_weights' + str(i_)])
            neigh_transformed = tf.reshape(neigh_transformed, [-1, nv_shape[1], self.transformed_dim])

            self_transformed = tf.matmul(self_vecs, self.vars['transform_weights' + str(i_)])

            # Neural Network
            self_transformed_tiled = tf.tile(tf.expand_dims(self_transformed, axis=1), [1, nv_shape[1], 1])
            concat_transformed = tf.concat([self_transformed_tiled, neigh_transformed], axis=-1)
            # Feedforward
            concat_shape = concat_transformed.get_shape().as_list()
            attention_out = tf.matmul(tf.reshape(concat_transformed, [-1, 2 * self.transformed_dim]),
                                      self.vars['attention_weights' + str(i_)])
            attention_out = tf.nn.leaky_relu(attention_out)
            attention_out = tf.reshape(attention_out, [-1, concat_shape[1]])

            # Apply soft-max row-wise
            attention_weights = tf.nn.softmax(attention_out, axis=1)
            attention_weights = tf.reshape(attention_weights, [-1, concat_shape[1], 1])

            # Get final hidden vectors
            neigh_weighted = neigh_transformed * attention_weights

            # Sum column wise (sum over samples)
            neigh_aggregated = tf.reduce_sum(neigh_weighted, axis=1)

            neigh_aggregated_heads.append(neigh_aggregated)

        concatenated_heads = tf.concat([tf.expand_dims(t, axis=1) for t in neigh_aggregated_heads], axis=1)
        final_neigh_aggregated = tf.reduce_mean(concatenated_heads, axis=1)

        # Apply Non-linearity
        final_neigh_aggregated = tf.nn.elu(final_neigh_aggregated)

        # Apply aggregation weights
        from_neighs = tf.matmul(final_neigh_aggregated, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars['self_weights'])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GINAggregator(Layer):
    """
        Graph Isomorphism Network "GIN" (https://arxiv.org/abs/1810.00826)
        Authors:
            Zahra Gharaee (zahra.gharaee@liu.se)
            Shreyas Kowshik (shreyaskowshik@iitkgp.ac.in)
        """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, model_size="small", **kwargs):
        super(GINAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = 128
        elif model_size == "big":
            hidden_dim = 256
        self.hidden_dim = hidden_dim

        self.learning_epsilon = True
        if self.learning_epsilon:
            # GIN-E
            # Learning epsilon with Initialization (0.5, 0.001)
            self.epsilon = tf.Variable(0.5)
        else:
            # GIN-0
            self.epsilon = 0

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.leaky_relu,
                                     dropout=dropout,
                                     BN=True, 
                                     sparse_inputs=False,
                                     logging=self.logging))
        
        for _ in range(1):
            self.mlp_layers.append(Dense(input_dim=hidden_dim,
                                         output_dim=hidden_dim,
                                         act=tf.nn.leaky_relu,
                                         dropout=dropout,
                                         BN=True,                                                                           
                                         sparse_inputs=False,
                                         logging=self.logging))
        
        self.mlp_layers.append(Dense(input_dim=hidden_dim,
                                     output_dim=output_dim,
                                     act=tf.nn.leaky_relu,
                                     dropout=dropout,
                                     BN=True,
                                     sparse_inputs=False,
                                     logging=self.logging))

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        """
        This function aggregates vectors of the neighbors using GIN.
        The obtained features followed by multiset function: MLP
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_sum = tf.reduce_sum(neigh_vecs, axis=1)

        output = tf.add_n([(1+self.epsilon)*self_vecs, neigh_sum])
        for layer in self.mlp_layers:
            output = layer(output)

        return output


class MeanAggregator(Layer):
    """
    "GraphSAGE-Mean" (https://arxiv.org/abs/1706.02216)
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        """
        This function aggregates vectors of the neighbors using mean
        It then uses different weights for the aggregated vectors and the self vector transforms them
        Concatenates the obtained representations followed by non-linearity

        neigh_vecs : 3D tensor of shape [num_nodes_whose_vectors_needed, num_samples_for_each_node, feature_size]
        """
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
       
        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class BatchGCNAggregator(Layer):
    """
        "BatchGCN"
        Not fully implemented and tested!
        Authors:
        Zahra Gharaee (zahra.gharaee@liu.se)
        Shreyas Kowshik (shreyaskowshik@iitkgp.ac.in)
        """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(BatchGCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.mlp = Dense(input_dim=2*output_dim,
                         output_dim=output_dim,
                         act=tf.nn.leaky_relu,
                         dropout=dropout,
                         sparse_inputs=False,
                         logging=self.logging)

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        output = self.act(output)

        # Do the batch aggregation step
        dims = tf.shape(neigh_vecs)
        batch_size = tf.cast(dims[0], tf.float32)

        all_mean = tf.reshape(tf.reduce_mean(output, axis=0), (1, self.output_dim))

        # Mean of remaining nodes in batch for each node
        remaining_aggregated = ((batch_size*all_mean) - output)/(batch_size - 1.0)
        final_cat = tf.concat([output, remaining_aggregated], axis=1)

        final_out = self.mlp(final_cat)

        return final_out


class GCNAggregator(Layer):
    """
        "GCN" (https://arxiv.org/abs/1609.02907)
        Aggregates via mean followed by matmul and non-linearity.
        Same matmul parameters are used self vector and neighbor vectors.
        """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class MaxPoolingAggregator(Layer):
    """
        "GraphSAGE-Maxpool" (https://arxiv.org/abs/1706.02216)
        Aggregates via max-pooling over MLP functions.
        """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        # for l in self.mlp_layers:
        #     h_reshaped = l(h_reshaped)
        # neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        # neigh_h = tf.reduce_mean(neigh_h, axis=1)
        #
        # from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        # from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        #

        # self_vecs, neigh_vecs = inputs
        #
        # neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        # self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)
        # neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class MeanPoolingAggregator(Layer):
    """
        "GraphSAGE-Meanpool" (https://arxiv.org/abs/1706.02216)
        Aggregates via mean-pooling over MLP functions.
        """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_mean(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """
        "GraphSAGE-Maxpool" (https://arxiv.org/abs/1706.02216)
        Aggregates via pooling over two MLP functions.
        """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                     output_dim=hidden_dim_1,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                     output_dim=hidden_dim_2,
                                     act=tf.nn.relu,
                                     dropout=dropout,
                                     sparse_inputs=False,
                                     logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class SeqAggregator(Layer):
    """
        "GraphSAGE-LSTM" (https://arxiv.org/abs/1706.02216)
        Aggregates via a standard LSTM.
        """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

