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
list_ncaps = [5,9] # [3,5,7,9,11] k/preference factors maybe 3 and 11 later
list_routit = [1,5,9] #
list_n_iter = [1,3]

eaf = False # extra article features
# eaf = True # extra article features

dataset_idx = 1
dataset = ["demo", "small", "large"][dataset_idx]
n_word = [20697, 35000, None][dataset_idx]
# already_loaded = False
already_loaded = True

args = set_parse_arguments()
if already_loaded:
   data = []
   for idx in range(10):
      data.append(np.load(f"Data/data_{str(eaf)}_{dataset}_{idx}.npy", allow_pickle=True))
else:
   data = load_data(args, extra_article_features=eaf, dataset=dataset)
   n_word = data[-1]
   print(f"n_word: {n_word}")
   data = data[:-1]
   for idx, dat in enumerate(data):
      np.save(f"Data/data_{str(eaf)}_{dataset}_{idx}", dat)


# for ncaps in list_ncaps:
#    args = set_parse_arguments(ncaps=ncaps)
#    train(args, data, show_loss, n_word)

# for routit in list_routit:
#    args = set_parse_arguments(routit=routit)
#    train(args, data, show_loss, n_word)

# for n_iter in list_n_iter[1:]:
   # print(f"n_iter: {n_iter}")
   # tf.compat.v1.reset_default_graph()
   # args = set_parse_arguments(n_iter=n_iter)
   # train(args, data, show_loss, n_word)

train(args, data, show_loss, n_word)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn import svm
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import f1_score, roc_auc_score

# train_data = data[0]
# eval_data = data[1]
# # print(len(train_data), len(eval_data))
# # 
# unit_model = svm.SVR

# unit_model = RandomForestRegressor
# param_dist = {
#    'n_estimators': [50, 100, 200, 400],
#    'max_depth': [10, 40, 70, 100, None],
#    'min_samples_split': [2, 5, 10],
#    'min_samples_leaf': [1, 2, 4],
#    'bootstrap': [True, False]
# }
# model = unit_model()
# rsh = RandomizedSearchCV(model, param_dist, random_state=42,n_jobs=1, verbose=2)
# rsh.fit(train_data[:, :2], train_data[:, 3])

# params = rsh.best_params_
# params = {}

# model2 = unit_model(**params)
# model2.fit(train_data[:, :2], train_data[:, 3])
# pred = model2.predict(eval_data[:, :2])

# print("f1: ", f1_score(eval_data[:, 3].astype(int), (pred > 0.5).astype(int)))
# print("roc_auc: ", roc_auc_score(eval_data[:, 3].astype(float), pred.astype(float)))

if show_time:
   print('time used: %d s' % (time.time() - t))
