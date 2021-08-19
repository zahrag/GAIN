from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
# import sklearn
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder

from codes.supervised_models import SupervisedGraphsage
from codes.models import SAGEInfo
from codes.minibatch import NodeMinibatchIterator
from codes.neigh_samplers import UniformNeighborSampler


flags = tf.app.flags
FLAGS = flags.FLAGS


def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro", zero_division=1), metrics.f1_score(y_true, y_pred, average="macro", zero_division=1)


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, log_dir=None, test_data=None):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map = train_data[4]

    # ---fix, non ordinal classes---
    classes_set = np.array(list(set(val for val in class_map.values())))
    enc = OrdinalEncoder()
    enc.fit(classes_set.reshape(-1, 1))
    class_map = {k: int(enc.transform(np.array(v).reshape(1, -1)).flatten()[0]) for k, v in class_map.items()}
    # ---------

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector at the bottom row
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    # print(len(context_pairs))

    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G, 
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      context_pairs=context_pairs
                                      )
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="mean",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=True,
                                    logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gcn",
                                    model_size=FLAGS.model_size,
                                    concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="seq",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=True,
                                    logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="maxpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=True,
                                    logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="meanpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=True,
                                    logging=True)

    elif FLAGS.model == 'attention':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="attn",
                                    sigmoid_loss=FLAGS.sigmoid,
                                    model_size=FLAGS.model_size,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=True,
                                    logging=True)

    elif FLAGS.model == 'gin':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gin",
                                    sigmoid_loss=FLAGS.sigmoid,
                                    model_size=FLAGS.model_size,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=False,
                                    logging=True)

    elif FLAGS.model == 'gain':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes,
                                    placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gain",
                                    sigmoid_loss=FLAGS.sigmoid,
                                    model_size=FLAGS.model_size,
                                    identity_dim=FLAGS.identity_dim,
                                    concat=False,
                                    logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # ### Init saver
    saver = tf.train.Saver(max_to_keep=8, keep_checkpoint_every_n_hours=1)

    # Train model
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    epoch_val_f1_scores = []
    
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    val_results_all = []
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle()

        print('Epoch: %04d' % (epoch + 1), 'learning rate: %0.2e' % FLAGS.learning_rate)

        iter = 0
        epoch_val_costs.append(0)
        val_cost_ = []
        val_f1_mic_ = []
        val_f1_mac_ = []
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged,
                             model.opt_op,
                             model.loss,
                             model.preds],
                            feed_dict={**feed_dict})
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac), 
                      "time=", "{:.5f}".format(avg_time))
 
            iter += 1
            total_steps += 1
            val_cost_.append(val_cost)
            val_f1_mic_.append(val_f1_mic)
            val_f1_mac_.append(val_f1_mac)

            if total_steps > FLAGS.max_total_steps:
                break

        val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.batch_size)
        epoch_val_f1_scores.append(val_f1_mac)

        if total_steps > FLAGS.max_total_steps:
            break

        # Save embeddings every N epochs
        if epoch % 10 == 0:
            # print("-------------train_results", all_tr_res)
            # print("-------------validation_results", all_val_res)

            if FLAGS.val_analysis:
                print(
                    "val_loss=", "{:.5f}".format(sum(val_cost_) / len(val_cost_)),
                    "val_f1_mic_=", "{:.5f}".format(sum(val_f1_mic_) / len(val_f1_mic_)),
                    "val_f1_mac_=", "{:.5f}".format(sum(val_f1_mac_) / len(val_f1_mac_)),
                    # exponential moving average
                )
                val_results_all.append([sum(val_cost_) / len(val_cost_),
                                        sum(val_f1_mic_) / len(val_f1_mic_),
                                        sum(val_f1_mac_) / len(val_f1_mac_)]
                                       )

                with open(log_dir + "eval_val_res" + ".txt", 'w') as f:
                    f.writelines(','.join(str(j) for j in val) + '\n' for val in val_results_all)

    print("Optimization Finished!")
    sess.run(val_adj_info.op)
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
    print("Full validation stats:",
          "loss=", "{:.5f}".format(val_cost),
          "f1_micro=", "{:.5f}".format(val_f1_mic),
          "f1_macro=", "{:.5f}".format(val_f1_mac),
          "time=", "{:.5f}".format(duration))
    with open(log_dir + "val_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
                format(val_cost, val_f1_mic, val_f1_mac, duration))

    print("Writing test set stats to file (don't peak!)")
    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)
    with open(log_dir + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                format(val_cost, val_f1_mic, val_f1_mac))

