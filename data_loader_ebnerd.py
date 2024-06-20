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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# import tensorflow as tf
# check if GPU is available
# print("1", tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print("3",tf.test.gpu_device_name())

def catlist_to_idlist(df: pl.DataFrame, column_name: str):
    all_topics = set()
    for row in df[column_name]:
        for topic in row:
            if topic not in all_topics:
                all_topics.add(topic)
    all_topics = list(all_topics)
    return df.with_columns(df[column_name].apply(lambda x: [all_topics.index(topic) for topic in x])), len(all_topics)

def cat_to_id(df: pl.DataFrame, column_name: str):
    all_topics = df[column_name].unique().to_list()
    return df.with_columns(df[column_name].apply(lambda x: all_topics.index(x))), len(all_topics)

    
def create_data(json_history, len_news):
    data = []
    for user in range(len(json_history)):
        data.append([user, json_history[user]['article_id_fixed'][-1], None, 1])
        read_news = [x for x in json_history[user]['article_id_fixed']]
        negative = random.sample(sorted(set(range(1, len_news + 1)) - set(read_news)), 1)[0] # Can't sample set, had to change this
        data.append([user, negative, None, 0])
    return data

def create_graph(json_history, len_news, args):
    t_user_news = defaultdict(list)
    t_news_user = defaultdict(list)
    user_news = np.zeros([len(json_history), args.news_neighbor], dtype=np.int32)
    news_user = np.zeros([len_news, args.user_neighbor], dtype=np.int32)
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
    return user_news, news_user
    
def load_data(args):
    PATH = Path("Data/ebnerd_demo")
    TEST_PATH = Path("Data/ebnerd_testset")
    
    # df_behaviors = pl.scan_parquet(PATH.joinpath(data_split, "behaviors.parquet"))
    
    df_history_train = pl.scan_parquet(PATH.joinpath("train", "history.parquet"))
    df_history_valid = pl.scan_parquet(PATH.joinpath("validation", "history.parquet"))
    #df_history_test = pl.scan_parquet(TEST_PATH.joinpath("test", "history.parquet"))
    

    json_history_train = json.loads(df_history_train.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).write_json(row_oriented=True))
    json_history_valid = json.loads(df_history_valid.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).write_json(row_oriented=True))
    json_history_test = json.loads(df_history_valid.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).write_json(row_oriented=True)) # load validation data as test data for now
    # json_history_test = json.loads(df_history_test.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).write_json(row_oriented=True)) # load validation data as test data for now

    relevant_columns = ['article_id', 'title', 'ner_clusters', 'entity_groups', 'article_type', 'premium', 'category', 'subcategory', 'sentiment_label', 'topics']
    title_column = 'title'
    ner_column = 'ner_clusters'
    ner_type_column = 'entity_groups'
    unnested_entity_columns = ['article_type', 'premium', 'category', 'sentiment_label']
    nested_entity_columns = ['topics', 'subcategory']

    df_articles = pl.scan_parquet(PATH.joinpath("articles.parquet"))
    df_articles = df_articles.collect().select(relevant_columns)
    df_articles = df_articles.with_columns(df_articles['title'].apply(lambda x: x.split()))

    # df_articles_test = pl.scan_parquet(TEST_PATH.joinpath("articles.parquet"))
    # df_articles_test = df_articles_test.collect().select(relevant_columns)
    # df_articles_test = df_articles_test.with_columns(df_articles_test['title'].apply(lambda x: x.split()))

    # df_articles = pl.concat([df_articles, df_articles_test]).unique(subset=['article_id'])

    column_n_unique = {}
    for column in nested_entity_columns + [title_column, ner_column, ner_type_column]:
        df_articles, length = catlist_to_idlist(df_articles, column)
        column_n_unique[column] = length

    for column in unnested_entity_columns:
        df_articles, length = cat_to_id(df_articles, column)
        column_n_unique[column] = length

    # Make list of entities for each article 
    all_entities = []
    for i in range(len(df_articles)):
        article_entities = []
        for column in ['ner_clusters', 'topics', 'subcategory']:
            article_entities.extend(list(df_articles[column][i]))
        for column in unnested_entity_columns:
            article_entities.append(df_articles[column][i])
        all_entities.append(article_entities)

    #all_entities = [list(df_articles['ner_clusters'][i]) + [df_articles['premium'][i]] + [df_articles['article_type'][i]] for i in range(len(df_articles))]
    
    # Count amount of NER groups
    all = set()
    for group_list in df_articles['entity_groups']: 
        all |= set(group_list)
    n_ner_groups = len(all)

    # Make list of entity types for each article
    all_groups = []
    for i in range(len(df_articles)):
        article_groups = list(df_articles['entity_groups'][i]) # Start with NER groups, which are different per entity
        article_groups.extend([n_ner_groups] * len(df_articles['topics'][i])) # Extend with representation for topic, using first unused number
        article_groups.extend([n_ner_groups + 1] * len(df_articles['subcategory'][i])) # Idem
        next_group_number = n_ner_groups + 2
        for column in unnested_entity_columns: # Append types for the single value columns
            article_groups.append(next_group_number)
            next_group_number += 1
        all_groups.append(article_groups)

    #all_groups = [list(df_articles['entity_groups'][i]) + [n_ner_groups + 1] + [n_ner_groups + 2] for i in range(len(df_articles))]


    # Mapping article ids to indices in article data
    art_id_to_idx = {}
    for row in range(len(df_articles)):
        id = df_articles[row]['article_id'][0]
        art_id_to_idx[id] = row
        

    # Remap article ids in history
    for user in range(len(json_history_train)):
        json_history_train[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history_train[user]['article_id_fixed']]

    for user in range(len(json_history_valid)):
        json_history_valid[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history_valid[user]['article_id_fixed']]
        
    for user in range(len(json_history_test)):
        json_history_test[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history_test[user]['article_id_fixed']]
        

    all_dates = []
    for user in range(len(json_history_train)):
        for article, time in zip(json_history_train[user]['article_id_fixed'], json_history_train[user]['impression_time_fixed']):
            all_dates.append(time)
            
    graph_cutoff = int(len(all) * (5/7))
    
    graph_cutoff_date = all_dates[graph_cutoff]
    
    graph_history = {}
    graph_history_icl_train = {}
    train_history = {}

    for user in range(len(json_history_train)):
        graph_history[user] = {'article_id_fixed': []}
        graph_history_icl_train[user] = {'article_id_fixed': []}
        train_history[user] = {'article_id_fixed': []}

        
        for article, time in zip(json_history_train[user]['article_id_fixed'], json_history_train[user]['impression_time_fixed']):
            if time < graph_cutoff_date:
                graph_history[user]['article_id_fixed'].append(article)
                graph_history_icl_train[user]['article_id_fixed'].append(article)
            else:
                train_history[user]['article_id_fixed'].append(article)
                graph_history_icl_train[user]['article_id_fixed'].append(article)   
        # make sure all users have at least one article in the training set
        if len(graph_history[user]['article_id_fixed']) == 0:
            graph_history[user]['article_id_fixed'].append(article)
        # if len(graph_history_icl_train[user]['article_id_fixed']) == 0:
        #     graph_history_icl_train[user]['article_id_fixed'].append(json_history_train[user]['article_id_fixed'][0])

    all = set()
    for entity_list in all_entities:
        all |= set(entity_list)
    n_entity = len(all)

    json_articles = json.loads(df_articles.write_json(row_oriented=True))
    
    news_title = []
    t_entity_news = defaultdict(list)
    entity_news = np.zeros([n_entity, args.news_neighbor], dtype=np.int64)
    news_entity = np.zeros([len(json_articles), args.entity_neighbor], dtype=np.int64)
    news_group = np.zeros([len(json_articles), args.entity_neighbor], dtype=np.int64)
    for article_id in range(len(json_articles)):

        if len(json_articles[article_id]['title']) <= args.title_len:
            json_articles[article_id]['title'].extend([0]*(args.title_len-len(json_articles[article_id]['title']))) #NB: in-place operation. Authors' code didn't work
        news_title.append(json_articles[article_id]['title'][:args.title_len])
        # sample entity neighbors of news
        entities = all_entities[article_id]
        groups = all_groups[article_id]
        n_neighbors = len(entities)

        if n_neighbors >= args.entity_neighbor:

            sampled_indices = np.random.choice(list(range(args.entity_neighbor)), size=args.entity_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.entity_neighbor,
                                                replace=True)
        news_entity[article_id] = np.array([entities[j] for j in sampled_indices])
        news_group[article_id] = np.array([groups[j] for j in sampled_indices])

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
    train_data = np.array(create_data(train_history, len_news))
    eval_data = np.array(create_data(json_history_valid, len_news))
    test_data = np.array(create_data(json_history_test, len_news))
    train_user_news, train_news_user = create_graph(graph_history, len_news, args)
    test_user_news, test_news_user = create_graph(graph_history_icl_train, len_news, args)

    return train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, news_title, news_entity, news_group