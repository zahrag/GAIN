from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from codes.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from codes.minibatch import EdgeMinibatchIterator
from codes.neigh_samplers import UniformNeighborSampler

flags = tf.app.flags
FLAGS = flags.FLAGS


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)


def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        # ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))


def construct_placeholders():
    # Define placeholders
    """
    For each minibatch, find all edges/random-walk co-ocurrences
    Take the first nodes of each edge and add to batch1
    Take the second nodes of each edge and add to batch2
    These are batch1 and batch2
    See minibatch.batch_feed_dict(batch_edges)
    """
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, log_dir=None, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    dfs_walks = None

    if not features is None:
        # pad with dummy zero vector at the bottom row
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None

    placeholders = construct_placeholders()
    # This gives an iterator over co-occurring edges' batch
    minibatch = EdgeMinibatchIterator(G,
                                      id_map,
                                      placeholders,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      num_neg_samples=FLAGS.neg_sample_size,
                                      context_pairs=context_pairs
                                      )
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)  # Matrix of size (V+1, max_graph_degree)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=True,
                                   logging=True)
    elif FLAGS.model == 'attention':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   aggregator_type="attn",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=True,
                                   logging=True)
    elif FLAGS.model == 'gin':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   aggregator_type="gin",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=False,
                                   logging=True)
    elif FLAGS.model == 'gain':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   aggregator_type="gain",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=False,
                                   logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   aggregator_type="gcn",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=False,
                                   logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   identity_dim=FLAGS.identity_dim,
                                   aggregator_type="seq",
                                   model_size=FLAGS.model_size,
                                   concat=True,
                                   logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   aggregator_type="maxpool",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=True,
                                   logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders,
                                   features,
                                   adj_info,
                                   minibatch.deg,
                                   layer_infos=layer_infos,
                                   dfs_walks=dfs_walks,
                                   aggregator_type="meanpool",
                                   model_size=FLAGS.model_size,
                                   identity_dim=FLAGS.identity_dim,
                                   concat=True,
                                   logging=True)
    elif FLAGS.model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                              minibatch.deg,
                              # 2x because graphsage uses concat
                              nodevec_dim=2 * FLAGS.dim_1,
                              lr=FLAGS.learning_rate)
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
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    val_results_all = []
    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()

        print('Epoch: %04d' % (epoch + 1), 'learning rate: %0.2e' % FLAGS.learning_rate)

        iter = 0
        epoch_val_costs.append(0)
        val_cost_ = []
        val_mrr_ = []
        shadow_mrr_ = []
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged,
                             model.opt_op,
                             model.loss,
                             model.ranks,
                             model.aff_all,
                             model.mrr,
                             model.outputs1],
                            feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr = outs[5]
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr  #
            else:
                train_shadow_mrr -= (1 - 0.99) * (train_shadow_mrr - train_mrr)

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sess.run(val_adj_info.op)
                val_cost, ranks, val_mrr, duration = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
                sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1 - 0.99) * (shadow_mrr - val_mrr)

            if total_steps % FLAGS.print_every == 0:
                summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                print_results_itr = False
                if print_results_itr:
                    print("Iter:", '%04d' % iter,
                          "train_loss=", "{:.5f}".format(train_cost),
                          "train_mrr=", "{:.5f}".format(train_mrr),
                          "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr),  # exponential moving average
                          "val_loss=", "{:.5f}".format(val_cost),
                          "val_mrr=", "{:.5f}".format(val_mrr),
                          "val_mrr_ema=", "{:.5f}".format(shadow_mrr),  # exponential moving average
                          "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1
            val_cost_.append(val_cost)
            val_mrr_.append(val_mrr)
            shadow_mrr_.append(shadow_mrr)

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

        # Save embeddings every N epochs
        if epoch % FLAGS.save_embeddings_epoch == 0:
            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir)
            if FLAGS.val_analysis:
                print(
                    "val_loss=", "{:.5f}".format(sum(val_cost_) / len(val_cost_)),
                    "val_mrr=", "{:.5f}".format(sum(val_mrr_) / len(val_mrr_)),
                    "val_mrr_ema=", "{:.5f}".format(sum(shadow_mrr_) / len(shadow_mrr_)),  # exponential moving average
                )
                val_results_all.append([sum(val_cost_) / len(val_cost_),
                                        sum(val_mrr_) / len(val_mrr_),
                                        sum(shadow_mrr_) / len(shadow_mrr_)])

                with open(log_dir + "eval_val_res" + ".txt", "w") as fp:
                    fp.write(val_results_all)

                with open(log_dir + "eval_val_res" + ".txt", 'w') as f:
                    f.writelines(','.join(str(j) for j in val) + '\n' for val in val_results_all)

    print("Optimization Finished!")

    if FLAGS.save_embeddings:

        sess.run(val_adj_info.op)
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir)

        if FLAGS.model == "n2v":
            # stopping the gradient for the already trained nodes
            train_ids = tf.constant(
                [[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                dtype=tf.int32)
            test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']],
                                   dtype=tf.int32)
            update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
            no_update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(train_ids))
            update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
            no_update_nodes = tf.stop_gradient(
                tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
            model.context_embeds = update_nodes + no_update_nodes
            sess.run(model.context_embeds)

            # run random walks
            nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
            start_time = time.time()
            pairs = None
            walk_time = time.time() - start_time

            test_minibatch = EdgeMinibatchIterator(G,
                                                   id_map,
                                                   placeholders, batch_size=FLAGS.batch_size,
                                                   max_degree=FLAGS.max_degree,
                                                   num_neg_samples=FLAGS.neg_sample_size,
                                                   context_pairs=pairs,
                                                   n2v_retrain=True,
                                                   fixed_n2v=True)

            start_time = time.time()
            print("Doing test training for n2v.")
            test_steps = 0
            for epoch in range(FLAGS.n2v_test_epochs):
                test_minibatch.shuffle()
                while not test_minibatch.end():
                    feed_dict = test_minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all,
                                     model.mrr, model.outputs1], feed_dict=feed_dict)
                    if test_steps % FLAGS.print_every == 0:
                        print("Iter:", '%04d' % test_steps,
                              "train_loss=", "{:.5f}".format(outs[1]),
                              "train_mrr=", "{:.5f}".format(outs[-2]))
                    test_steps += 1
            train_time = time.time() - start_time
            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir, mod="-test")
            print("Total time: ", train_time + walk_time)
            print("Walk time: ", walk_time)
            print("Train time: ", train_time)
