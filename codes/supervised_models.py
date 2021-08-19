import tensorflow as tf

import codes.models as models
import codes.layers as layers

from .aggregators import AttentionAggregator, GINAggregator, GAINAggregator, BatchGCNAggregator
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - learning_rate : A placeholder
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == "attn":
            self.aggregator_cls = AttentionAggregator
        elif aggregator_type == "gin":
            self.aggregator_cls = GINAggregator
        elif aggregator_type == "gain":
            self.aggregator_cls = GAINAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        # Apply aggregation:
        # generate aggregator output and aggregator function
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims,
                                                         num_samples, support_sizes1,
                                                         concat=self.concat, model_size=self.model_size
                                                         )
        dim_mult = 2 if self.concat else 1

        # Normalize aggregator output
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        # Apply Dense layer to the aggregator output to calculate node prediction [output_dim x Number_of_classes]
        # Initialize prediction function:
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                                      dropout=self.placeholders['dropout'],
                                      act=lambda x: x
                                      )
        # TF graph management
        # node_preds = outputs1 x layer_weights + layer_bias
        # layer_weights: [output_dim x Number_of_classes]
        # layer_weights: variables to be optimised
        self.node_preds = self.node_pred(self.outputs1)

        # Calculate loss function based on model variables
        self._loss()

        # Calculate gradients
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                                  for grad, var in grads_and_vars
                                  ]
        self.grad, _ = clipped_grads_and_vars[0]

        # Optimisation
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

        # Prediction: Apply a Sigmoid or Softmax to node_preds
        self.preds = self.predict()

    def _loss(self):

        # ### Weight decay loss (l2_loss):
        # to optimise aggregator variables
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # to optimise node prediction variables
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # ### Classification loss (sigmoid_cross_entropy_with_logits):
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
