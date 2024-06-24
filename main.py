import argparse
import numpy as np
import time
from data_loader_ebnerd import load_data
from train import train

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

def set_parse_arguments(ncaps=7, routit=7, n_iter=2):
   parser = argparse.ArgumentParser()

   parser.add_argument("--news_neighbor", type=int, default=30, help="the number of neighbors to be sampled")
   parser.add_argument("--entity_neighbor", type=int, default=40, help="the number of neighbors to be sampled")
   parser.add_argument("--user_neighbor", type=int, default=30, help="the number of neighbors to be sampled")
   parser.add_argument("--title_len", type=int, default=10, help="the max length of title")
   parser.add_argument("--ratio", type=float, default=0.2, help="the ratio of train data")
   parser.add_argument('--dataset', type=str, default='ten_week', help='which dataset to use')
   parser.add_argument('--session_len', type=int, default=10, help='the max length of session')
   parser.add_argument('--aggregator', type=str, default='neighbor', help='which aggregator to use')
   parser.add_argument('--n_epochs', type=int, default=3, help='the number of epochs')
   parser.add_argument('--user_dim', type=int, default=128, help='dimension of user and entity embeddings')
   parser.add_argument('--cnn_out_size', type=int, default=128, help='dimension of cnn output')
   parser.add_argument('--n_iter', type=int, default=n_iter, help='number of iterations when computing entity representation')
   parser.add_argument('--batch_size', type=int, default=32, help='batch size')
   parser.add_argument('--l2_weight', type=float, default=5e-3, help='weight of l2 regularization')
   parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')  #3e-4
   parser.add_argument('--save_path', type=str, default="./Data/1week/hop2/version1/", help='model save path')
   parser.add_argument('--test', type=int, default=0, help='test')
   parser.add_argument('--use_group', type=int, default=1, help='whether use group')
   parser.add_argument('--n_filters', type=int, default=64, help='number of filters for each size in KCNN')
   parser.add_argument('--filter_sizes', type=int, default=[2, 3], nargs='+',
                     help='list of filter sizes, e.g., --filter_sizes 2 3')
   parser.add_argument('--ncaps', type=int, default=ncaps,
                     help='Maximum number of capsules per layer.')
   parser.add_argument('--dcaps', type=int, default=0, help='Decrease this number of capsules per layer.')
   parser.add_argument('--nhidden', type=int, default=16,
                           help='Number of hidden units per capsule.')
   parser.add_argument('--routit', type=int, default=routit,
                     help='Number of iterations when routing.')
   parser.add_argument('--balance', type=float, default=0.004, help='learning rate')  #3e-4
   parser.add_argument('--version', type=int, default=0,
                           help='Different version under the same set')
   return parser.parse_args()

show_loss = True
show_time = True

t = time.time()
list_ncaps = [5,7,9] # [3,5,7,9,11] k/preference factors maybe 3 and 11 later
list_routit = [1,5,9] #
list_n_iter = [1,2,3]

eaf = False # extra article features
# eaf = True # extra article features

dataset = ["demo", "small", "large"][1]

already_loaded = False
# already_loaded = True

args = set_parse_arguments()
if already_loaded:
   data = []
   for idx in range(10):
      data.append(np.load(f"Data/data_{str(eaf)}_{dataset}_{idx}.npy", allow_pickle=True))
else:
   data = load_data(args, extra_article_features=eaf, dataset=dataset)

   for idx, dat in enumerate(data):
      np.save(f"Data/data_{str(eaf)}_{dataset}_{idx}", dat)


# for ncaps in list_ncaps:
#    args = set_parse_arguments(ncaps=ncaps)
#    train(args, data, show_loss)

# for routit in list_routit:
#    args = set_parse_arguments(routit=routit)
#    train(args, data, show_loss)

# for n_iter in list_n_iter:
#    args = set_parse_arguments(n_iter=n_iter)
#    train(args, data, show_loss)

train(args, data, show_loss)

if show_time:
   print('time used: %d s' % (time.time() - t))
