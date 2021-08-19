import os
import time
import datetime
import tensorflow as tf
import numpy as np

from codes.load_input import load_data
from codes.unsupervised_train import train

''' 
    Authors: Zahra Gharaee (zahra.gharaee@liu.se)

    This script contains codes to apply the settings required for the exhaustive grid search implementation of the 
    unsupervised experiments presented by GAIN paper.

    '''

# random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Settings
dir_data_transductive = '../graph_data/osm_transductive/'
dir_data_inductive = '../graph_data/osm_inductive/'

data_dirs = [dir_data_transductive,  # 0 transductive
             dir_data_inductive,     # 1 inductive
             ]

prefixes = ['linkoping-osm',         # 0
            'sweden-osm',            # 1
            ]

agg_funcs = ['graphsage_mean',       # 0
             'gcn',                  # 1
             'graphsage_meanpool',   # 2
             'graphsage_maxpool',    # 3
             'graphsage_seq',        # 4
             'attention',            # 5
             'gin',                  # 6
             'gain',                 # 7
             ]

batch_sizes = [2*512,                # 0 transductive ,
               4*512,                # 1 inductive
               ]

epochs = [500,                       # 0 supervised
          1000,                      # 1 unsupervised
          ]

# Dataset & Prefix
datadir = data_dirs[1]
prefix = prefixes[1]

# Aggregator
agg = agg_funcs[0]

# Dimension (concatenation has to be considered)
if agg == 'gcn' or agg == 'gin' or agg == 'gain':
    dimensions = [64, 128, 256]
else:
    dimensions = [32, 64, 128]

dim = dimensions[0]

# Learning-Rate
learning_rates = [2e-8, 2e-7, 2e-6, 2e-5]
l_r = learning_rates[0]

# Batch-Size
batch_size = batch_sizes[0]
if datadir == dir_data_inductive:
    batch_size = batch_sizes[1]

# Walk-Type
rand_context = True
walk_types = ['rand_edges',          # 0
              'rand_bfs_walks',      # 1
              'rand_bfs_dfs_walks',  # 2
              ]
walk_type = walk_types[0]
if rand_context:
    walk_type = walk_types[2]


# Epoch
epoch = epochs[1]


# Input Dataset
flags.DEFINE_string('data_dir', datadir, 'base directory for input')
flags.DEFINE_string('train_prefix', prefix,
                    'name of the object file that stores the training data (Inductive or Transductive).')

# walk
flags.DEFINE_boolean('random_context', rand_context, 'Whether to use random context or direct edges')
flags.DEFINE_string('walk_type', walk_type, 'How to create neighborhood?')
flags.DEFINE_integer('BFS_num', 50, 'number of BFS walks (local neighborhood in paper)')
flags.DEFINE_integer('BFS_len', 5, 'length of BFS walks (local neighborhood in paper)')
flags.DEFINE_integer('DFS_num', 50, 'number of DFS walks (global neighborhood in paper)')
flags.DEFINE_integer('DFS_len', 10, 'length of DFS walks (global neighborhood in paper)')

# core parameters
flags.DEFINE_string('model', agg, 'approach to aggregation.')
flags.DEFINE_string('lr_scheduler', "adam", 'optimizer.')
flags.DEFINE_float('learning_rate', l_r, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")

# left to default values in main experiments
flags.DEFINE_integer('epochs', epoch, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.1, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 9, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 3, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', dim, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', dim, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('neg_sample_size', 12, 'number of negative samples')
flags.DEFINE_integer('batch_size', batch_size, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_integer('classif_n_iter', 100, 'how many times to run the classification evaluation on embeddings')

# logging, saving, validation settings etc.
flags.DEFINE_boolean('val_analysis', False, 'whether to alanyse performance of validation set')
flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_integer('save_embeddings_epoch', 10, "how often to save embeddings.")
flags.DEFINE_boolean('save_checkpoints', True, 'whether to save model checkpoints')
flags.DEFINE_integer('save_checkpoints_every', 100, 'How often to save model checkpoints')
flags.DEFINE_integer('validate_iter', 200, "how often to run a validation minibatch.")  # 10
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
flags.DEFINE_string('timestamp', datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"),
                    'current date and time (local)')

# flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')

# os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
GPU_MEM_FRACTION = 0.8


def log_dir():

    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-1]
    log_dir += "/{timestamp:s}_{model:s}_{lr:.2e}/".format(
        timestamp=FLAGS.timestamp,
        model=FLAGS.model,
        lr=FLAGS.learning_rate)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir


def save_configs(log_dir=None):

    config_file = "configs"
    info = 'epochs:{}\ndropout:{}\nweight_decay:{}\nsample_1:{}\nsample_2:{}\n'.format(FLAGS.epochs,
                                                                                       FLAGS.dropout,
                                                                                       FLAGS.weight_decay,
                                                                                       FLAGS.samples_1,
                                                                                       FLAGS.samples_2)

    info += 'dim_1:{}\ndim_2:{}\nneg_sam_size:{}\nbatch_size:{}\nval_batch_size:{}\n'.format(FLAGS.dim_1,
                                                                                             FLAGS.dim_2,
                                                                                             FLAGS.neg_sample_size,
                                                                                             FLAGS.batch_size,
                                                                                             FLAGS.validate_batch_size)

    info += 'Model:{}\nlr_Scheduler:{}\nInitial_Leraning_Rate:{}\nModel_Size:{}\nPrefix:{}\n'.format(FLAGS.model,
                                                                                                     FLAGS.lr_scheduler,
                                                                                                     FLAGS.learning_rate,
                                                                                                     FLAGS.model_size,
                                                                                                     FLAGS.train_prefix)

    info += 'rand_context:{}\nwalk_type:{}\nBFS_num:{}\nBFS_len:{}\nDFS_num:{}\nDFS_len:{}\n'.format(FLAGS.random_context,
                                                                                                     FLAGS.walk_type,
                                                                                                     FLAGS.BFS_num,
                                                                                                     FLAGS.BFS_len,
                                                                                                     FLAGS.DFS_num,
                                                                                                     FLAGS.DFS_len)

    info += 'Start_Date&Time:{}'.format(FLAGS.timestamp)

    with open(log_dir + config_file + ".txt", "w") as fp:
        fp.write(info)


def main():

    start_time = time.asctime(time.localtime(time.time()))
    print('start time:', start_time)

    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix, FLAGS.data_dir,
                           walk_type=FLAGS.walk_type,
                           dfs_num_len=[FLAGS.DFS_num, FLAGS.DFS_len])

    print('Save configurations..')
    save_configs(log_dir=log_dir())

    print("Done loading training data..")
    train(train_data, log_dir=log_dir())


if __name__ == '__main__':
    tf.app.run()

