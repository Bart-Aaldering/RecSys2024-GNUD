from pathlib import Path
import polars as pl
from typing import List
import pandas as pd
import numpy as np
from collections import defaultdict
import random
import time
import datetime
import argparse
import json
from train import train
import os





def trans_time(linux_time, utc_time):
    UTC_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
    utcTime = datetime.datetime.strptime(utc_time, UTC_FORMAT)
    ans_time = time.mktime(utcTime.timetuple())

    return linux_time - ans_time

def dataset_split(data, args):
    print('splitting dataset ...')

    eval_ratio = (1 - args.ratio)/2
    test_ratio = (1 - args.ratio)/2
    n_samples = data.shape[0]

    eval_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * eval_ratio), replace=False)
    left = set(range(n_samples)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    print(n_samples,len(eval_indices),len(test_indices),len(train_indices))
    train_data = data[train_indices]
    eval_data = data[eval_indices]
    test_data = data[test_indices]

    return train_data, eval_data, test_data

def get_minutes(timedelta):
    try:
        return timedelta.seconds // 60
    except Exception:
        print("Error in get_minutes")
        return 'n/a'

def datetime_to_int(df: pl.DataFrame, column_names: List[str]):
    for column_name in column_names:
        df = df.with_columns(((df[column_name] - df[column_name].min()).map_elements(get_minutes)))
    return df

def catlist_to_idlist(df: pl.DataFrame, column_name: str):
    all_topics = set()
    for row in df[column_name]:
        for topic in row:
            if topic not in all_topics:
                all_topics.add(topic)
    all_topics = list(all_topics)
    return df.with_columns(df[column_name].map_elements(lambda x: [all_topics.index(topic) for topic in x])), len(all_topics)

def cat_to_id(df: pl.DataFrame, column_name: str):
    all_topics = df[column_name].unique().to_list()
    return df.with_columns(df[column_name].map_elements(lambda x: all_topics.index(x))), len(all_topics)

#def art_to_idx(df: pl.DataFrame, id: str):
    

PATH = Path("Data/ebnerd_demo")
# ARTICLES_PATH = Path("Data/ebnerd_demo")
data_split = "train"


# df_behaviors = pl.scan_parquet(PATH.joinpath(data_split, "behaviors.parquet"))
df_history = pl.scan_parquet(PATH.joinpath(data_split, "history.parquet"))
df_articles = pl.scan_parquet(PATH.joinpath("articles.parquet"))

df_history = df_history.collect().select(["user_id", "article_id_fixed"])
json_history = json.loads(df_history.write_json(row_oriented=True))

# df_behaviors = df_behaviors.select(["user_id", "article_id"])
# json_behaviors = json.loads(df_behaviors.collect().write_json(row_oriented=True))



# relevant_columns = ['last_modified_time', 'premium', 'published_time', 'image_ids', 
#                     'article_type', 'ner_clusters', 'entity_groups', 'topics', 'category', 
#                     'subcategory', 'total_inviews', 'total_pageviews', 'total_read_time', 
#                     'sentiment_score', 'sentiment_label']
relevant_columns = ['article_id', 'title', 'ner_clusters', 'entity_groups', 'article_type', 'premium']
entity_columns = ['ner_clusters', 'entity_groups', 'article_type', 'premium']
nested_columns = ['title', 'ner_clusters', 'entity_groups']
# df_articles = df_articles.collect().select(relevant_columns)
df_articles = df_articles.collect().select(relevant_columns)

# read_time_fixed impression_time_fixed scroll_percentage_fixed
# nested_columns = ['ner_clusters', 'entity_groups', 'topics', 'subcategory', 'image_ids']

# df_articles = datetime_to_int(df_articles, ['last_modified_time', 'published_time'])
df_articles = df_articles.with_columns(df_articles['title'].map_elements(lambda x: x.split()))
column_n_unique = {}
for column in nested_columns:
    df_articles, length = catlist_to_idlist(df_articles, column)
    column_n_unique[column] = length

for column in ['article_type', 'premium']:
    df_articles, length = cat_to_id(df_articles, column)
    column_n_unique[column] = length

all_entities = [list(df_articles['ner_clusters'][i]) + [df_articles['premium'][i]] + [df_articles['article_type'][i]] for i in range(len(df_articles))]
all = set()
for group_list in df_articles['entity_groups']: # Get amount of groups that we have already
    all |= set(group_list)
n_ner_groups = len(all)
all_groups = [list(df_articles['entity_groups'][i]) + [n_ner_groups + 1] + [n_ner_groups + 2] for i in range(len(df_articles))]

# Mapping article ids to indices in article data
art_id_to_idx = {}
for row in range(len(df_articles)):
    id = df_articles[row]['article_id'][0]
    art_id_to_idx[id] = row

#df_history = df_history.with_columns(df_history['article_id'].map_elements(lambda x: [art_id_to_idx[article_id] for article_id in x]))
# Remap article ids in history
for user in range(len(json_history)):
    json_history[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history[user]['article_id_fixed']]

# pl.Series("entitys", all_entities)
# pl.Series("groups", all_groups)

# start_of_id = 0
# for column in entity_columns:
#     df_articles = df_articles.with_columns(df_articles[column] + start_of_id)
#     start_of_id += df_articles[column].n_unique()

# for column in entity_columns:
#     print(column)
#     print(df_articles[column].min())
#     print(df_articles[column].max())

all = set()
for entity_list in all_entities:
    all |= set(entity_list)
n_entity = len(all)
# n_entity = len(set([entities for entities in all_entities]))



json_articles = json.loads(df_articles.write_json(row_oriented=True))


def main(args):
    news_title = []
    t_entity_news = defaultdict(list)
    entity_news = np.zeros([1 + n_entity, args.news_neighbor], dtype=np.int64)
    news_entity = np.zeros([1 + len(json_articles), args.entity_neighbor], dtype=np.int64)
    for article_id in range(len(json_articles)):

        if len(json_articles[article_id]['title']) <= args.title_len:
            json_articles[article_id]['title'].extend([0]*(args.title_len-len(json_articles[article_id]['title']))) #NB: in-place operation. Authors' code didn't work
        news_title.append(json_articles[article_id]['title'][:args.title_len])
        # sample entity neighbors of news
        entities = all_entities[article_id]
        n_neighbors = len(entities)
        if n_neighbors >= args.entity_neighbor:

            sampled_indices = np.random.choice(list(range(args.entity_neighbor)), size=args.entity_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.entity_neighbor,
                                                replace=True)
        news_entity[article_id] = np.array([entities[j] for j in sampled_indices])

        for e in entities:
            t_entity_news[e].append(article_id)

    # sample news neighbors of entity
    for j in range(len(t_entity_news)):
        n_neighbors = len(t_entity_news[j])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=True)
        entity_news[j] = np.array([t_entity_news[j][k] for k in sampled_indices])
    news_title = np.array(news_title)
    
    len_news = len(json_articles)
    t_user_news = defaultdict(list)
    t_news_user = defaultdict(list)
    user_news = np.zeros([1+len(json_history), args.news_neighbor], dtype=np.int32)
    news_user = np.zeros([1+len_news, args.user_neighbor], dtype=np.int32)
    data = []
    for user in range(len(json_history)):
        for article_id in range(len(json_history[user]['article_id_fixed'])):
            t_user_news[user].append(json_history[user]['article_id_fixed'][article_id])
            t_news_user[json_history[user]['article_id_fixed'][article_id]].append(user)

        # sample news neighbors of user
        n_neighbors = len(t_user_news[int(user)])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=True)
        user_news[int(user)] = np.array([t_user_news[int(user)][i] for i in sampled_indices])



        # t1 = trans_time(json_history[user]['time'][-1], json_history[user]['publishtime'][-1])
        data.append([user, json_history[user]['article_id_fixed'][-1], None, 1])
        

        read_news = [x for x in json_history[user]['article_id_fixed']]
        negative = random.sample(sorted(set(range(1, len_news + 1)) - set(read_news)), 1)[0] # Can't sample set, had to change this
        # t2 = trans_time(json_articles[negtive]['time'], json_articles[negtive]['publishtime'])
        data.append([user, negative, None, 0])
        
    # sample user neighbors of news
    for article in t_news_user.keys():
        n_neighbors = len(t_news_user[article])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor,
                                                replace=True)
        news_user[article] = np.array([t_news_user[article][j] for j in sampled_indices])

    # dataset split
    train_data, eval_data, test_data = dataset_split(np.array(data), args)


    # eval_indices = np.random.choice(list(range(test_data.shape[0])), size=int(test_data.shape[0] * 0.2), replace=False)
    # test_indices = list(set(range(test_data.shape[0])) - set(eval_indices))
    # eval_data = test_data[eval_indices]
    # test_data = test_data[test_indices]

    # np.random.shuffle(test_data)
    # np.random.shuffle(train_data)
    # l = int(len(test_data) * 0.1)

    # eval_data = test_data[:l]
    # test_data = test_data[l:]

    # cutoff_user_news = len(user_news)*0.8
    # cutoff_news_user = len(news_user)*0.8
    # train_user_news = user_news[:cutoff_user_news]
    # train_news_user = news_user[:cutoff_news_user]
    # test_user_news = user_news[cutoff_user_news:]
    # test_news_user = news_user[cutoff_news_user:]

    train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, news_title, news_entity, news_group


parser = argparse.ArgumentParser()
parser.add_argument("--news_neighbor", type=int, default=30, help="the number of neighbors to be sampled")
parser.add_argument("--entity_neighbor", type=int, default=30, help="the number of neighbors to be sampled")
parser.add_argument("--user_neighbor", type=int, default=30, help="the number of neighbors to be sampled")
parser.add_argument("--title_len", type=int, default=20, help="the max length of title")
parser.add_argument("--ratio", type=float, default=0.2, help="the ratio of train data")
args = parser.parse_args()

data = main(args)
train(args, data, True)