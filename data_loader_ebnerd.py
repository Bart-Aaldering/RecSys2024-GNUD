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

def catlist_to_idlist(df: pl.DataFrame, column_name: str, idx_offset: int = 0):
    all_topics = set()
    for row in df[column_name]:
        for topic in row:
            if topic not in all_topics:
                all_topics.add(topic)
    all_topics = list(all_topics)
    return df.with_columns(df[column_name].apply(lambda x: [all_topics.index(topic) + idx_offset for topic in x])), len(all_topics)

def cat_to_id(df: pl.DataFrame, column_name: str, idx_offset: int = 0):
    all_topics = df[column_name].unique().to_list()
    return df.with_columns(df[column_name].apply(lambda x: all_topics.index(x) + idx_offset)), len(all_topics)

def create_data(json_history, len_news):
    data = []
    for user in range(len(json_history)):
        if len(json_history[user]['article_id_fixed']) == 0:
            continue
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
    
def load_data(args, extra_article_features = False, dataset="demo"):
    PATH = Path(f"Data/ebnerd_{dataset}")
    TEST_PATH = Path("Data/ebnerd_testset")
    
    # df_behaviors = pl.scan_parquet(PATH.joinpath(data_split, "behaviors.parquet"))
    times = []
    old_time = time.time()
    print("Loading data 1")
    df_history_train = pl.scan_parquet(PATH.joinpath("train", "history.parquet"))
    df_history_valid = pl.scan_parquet(PATH.joinpath("validation", "history.parquet"))
    # df_history_test = pl.scan_parquet(TEST_PATH.joinpath("test", "history.parquet"))
    
    json_history_train = df_history_train.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).to_dicts()
    json_history_valid = df_history_valid.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).to_dicts()
    json_history_test = df_history_valid.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).to_dicts() # load validation data as test data for now
    # json_history_test = df_history_test.collect().select(["user_id", "article_id_fixed", "impression_time_fixed"]).to_dicts() # load validation data as test data for now

    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 2")
    # relevant_columns = ['article_id', 'title', 'ner_clusters', 'entity_groups', 'article_type', 'premium', 'category', 'subcategory', 'sentiment_label', 'topics']
    title_column = 'title'
    ner_type_column = 'entity_groups'
    if extra_article_features:
        nested_entity_columns = ['ner_clusters', 'topics', 'subcategory']
        unnested_entity_columns = ['article_type', 'premium', 'category', 'sentiment_label']
    else:
        nested_entity_columns = ['ner_clusters']
        unnested_entity_columns = []
    relevant_columns = ['article_id'] + [title_column] + [ner_type_column] + nested_entity_columns + unnested_entity_columns

    df_articles = pl.scan_parquet(PATH.joinpath("articles.parquet"))
    df_articles = df_articles.collect().select(relevant_columns)
    df_articles = df_articles.with_columns(df_articles['title'].apply(lambda x: x.split()))

    # df_articles_test = pl.scan_parquet(TEST_PATH.joinpath("articles.parquet"))
    # df_articles_test = df_articles_test.collect().select(relevant_columns)
    # df_articles_test = df_articles_test.with_columns(df_articles_test['title'].apply(lambda x: x.split()))

    # df_articles = pl.concat([df_articles, df_articles_test]).unique(subset=['article_id'])
    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 3")
    # convert categories to integers
    for column in [title_column, ner_type_column]:
        df_articles, length = catlist_to_idlist(df_articles, column)
    n_ner_groups = length
    idx_offset = 0
    for column in nested_entity_columns:
        df_articles, length = catlist_to_idlist(df_articles, column, idx_offset)
        idx_offset += length

    for column in unnested_entity_columns:
        df_articles, length = cat_to_id(df_articles, column, idx_offset)
        idx_offset += length
    n_entity = idx_offset

    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 4")
    # Make list of entities and entity groups for each article 
    all_entities = []
    all_groups = []
    for i in range(len(df_articles)):
        article_entities = []
        for column in nested_entity_columns:
            article_entities.extend(list(df_articles[column][i]))
        for column in unnested_entity_columns:
            article_entities.append(df_articles[column][i])
        if len(article_entities) == 0:
            article_entities.append(0)
        all_entities.append(article_entities)
    
        article_groups = list(df_articles[ner_type_column][i]) # Start with NER groups, which are different per entity
        offset = 0
        for column in nested_entity_columns:
            if column == 'ner_clusters':
                continue
            article_groups.extend([n_ner_groups + offset] * len(df_articles[column][i])) # Extend with representation for topic, using first unused number
            offset += 1

        for column in unnested_entity_columns: # Append types for the single value columns
            article_groups.append(offset)
            offset += 1
        if len(article_groups) == 0:
            article_groups.append(0)
        all_groups.append(article_groups)
    print("number of different entity types:", n_ner_groups+offset)
    # Mapping article ids to indices in article data
    art_id_to_idx = {}
    for row in range(len(df_articles)):
        id = df_articles[row]['article_id'][0]
        art_id_to_idx[id] = row

    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 6")
    # Remap article ids in history
    for user in range(len(json_history_train)):
        json_history_train[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history_train[user]['article_id_fixed']]
    for user in range(len(json_history_valid)):
        json_history_valid[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history_valid[user]['article_id_fixed']]
    for user in range(len(json_history_test)):
        json_history_test[user]['article_id_fixed'] = [art_id_to_idx[id] for id in json_history_test[user]['article_id_fixed']]
        
    # calculate cutoff date for splitting training data in train and graph data
    all_dates = []
    for user in range(len(json_history_train)):
        for date in json_history_train[user]['impression_time_fixed']:
            all_dates.append(date) 
    all_dates.sort()
    graph_cutoff_date = all_dates[int(len(all_dates) * (5/6))]
    
    graph_history = {}
    graph_history_icl_train = {}
    train_history = {}

    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 7")
    for user in range(len(json_history_train)):
        graph_history[user] = {'article_id_fixed': []}
        graph_history_icl_train[user] = {'article_id_fixed': []}
        train_history[user] = {'article_id_fixed': []}

        for article, date in zip(json_history_train[user]['article_id_fixed'], json_history_train[user]['impression_time_fixed']):
            if date < graph_cutoff_date:
                graph_history[user]['article_id_fixed'].append(article)
                graph_history_icl_train[user]['article_id_fixed'].append(article)
            else:
                train_history[user]['article_id_fixed'].append(article)
                graph_history_icl_train[user]['article_id_fixed'].append(article)

        # make sure all users have at least one article in the training set
        if len(graph_history[user]['article_id_fixed']) == 0:
            graph_history[user]['article_id_fixed'].append(article)
        if len(train_history[user]['article_id_fixed']) == 0:
            train_history[user]['article_id_fixed'].append(article)
        if len(graph_history_icl_train[user]['article_id_fixed']) == 0:
            graph_history_icl_train[user]['article_id_fixed'].append(article)

    json_articles = df_articles.to_dicts()

    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 8")
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
    
    times.append(time.time() - old_time)
    old_time = time.time()
    print("Loading data 9")
    len_news = len(json_articles)
    train_data = np.array(create_data(train_history, len_news))
    eval_data = np.array(create_data(json_history_valid, len_news))
    test_data = np.array(create_data(json_history_test, len_news))
    train_user_news, train_news_user = create_graph(graph_history, len_news, args)
    test_user_news, test_news_user = create_graph(graph_history_icl_train, len_news, args)
    times.append(time.time() - old_time)
    print("times:", times)
    return train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, news_title, news_entity, news_group