from pathlib import Path
import polars as pl
from typing import List
import pandas as pd





PATH = Path("Data/ebnerd_demo")
# ARTICLES_PATH = Path("Data/ebnerd_demo")
data_split = "train"

df_behaviors = pl.scan_parquet(PATH.joinpath(data_split, "behaviors.parquet"))
# df_history = pl.scan_parquet(PATH.joinpath(data_split, "history.parquet"))
df_articles = pl.scan_parquet(PATH.joinpath("articles.parquet"))


df_articles = df_articles.head(2000).collect()

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

def catlist_to_idlist(df: pl.DataFrame, column_names: List[str]):
    for column_name in column_names:
        df = df.with_columns(df[column_name].fill_null([]))
        all_topics = set()
        for row in df[column_name]:
            for topic in row:
                if topic not in all_topics:
                    all_topics.add(topic)
        all_topics = list(all_topics)
        df = df.with_columns(df[column_name].map_elements(lambda x: [all_topics.index(topic) for topic in x]))
    return df

relevant_columns = ['last_modified_time', 'premium', 'published_time', 'image_ids', 
                    'article_type', 'ner_clusters', 'entity_groups', 'topics', 'category', 
                    'subcategory', 'total_inviews', 'total_pageviews', 'total_read_time', 
                    'sentiment_score', 'sentiment_label']
print(df_articles.schema)
df_articles = df_articles.select(relevant_columns)
# print(df_articles)

nested_columns = ['ner_clusters', 'entity_groups', 'topics', 'subcategory', 'image_ids']
df_articles = datetime_to_int(df_articles, ['last_modified_time', 'published_time'])
df_articles = catlist_to_idlist(df_articles, nested_columns)
n_uniq = []
for column in df_articles.columns:
    # print(column)
    n_uniq.append(df_articles[column].n_unique())
print(n_uniq)
print(sum(n_uniq))

normal_columns = list(set(relevant_columns) - set(nested_columns))
for column in normal_columns:
    print('column', column)
    
    df_articles = df_articles.with_columns(pl.from_numpy(pd.Categorical(df_articles[column].to_pandas()).codes, schema=[column]))
    # print(df_articles[column].unique().list)
    # df_articles = df_articles.with_columns(df_articles[column].cast(pl.Int64))
    print(df_articles[column].min())
    print(df_articles[column].max())






def main():
    news_title = []
    n_entity = 69473
    t_entity_news = defaultdict(list)
    entity_news = np.zeros([1 + n_entity, args.news_neighbor], dtype=np.int64)
    news_entity = np.zeros([1 + len(news), args.entity_neighbor], dtype=np.int64)
    for i in range(1, len(news) + 1):

        if len(news[str(i)]['title']) <= args.title_len:
            news_title.append(news[str(i)]['title'].extend([0]*(args.title_len-len(news[str(i)]['title']))))
        else:
            news_title.append(news[str(i)]['title'][:args.title_len])
        # sample entity neighbors of news
        n_neighbors = len(news[str(i)]['entity'])
        if n_neighbors >= args.entity_neighbor:

            sampled_indices = np.random.choice(list(range(args.entity_neighbor)), size=args.entity_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.entity_neighbor,
                                                replace=True)
        news_entity[i] = np.array([news[str(i)]['entity'][j] for j in sampled_indices])

        for e in news[str(i)]['entity']:
            t_entity_news[e].append(i)

    # sample news neighbors of entity
    for j in range(1, len(t_entity_news) + 1):
        n_neighbors = len(t_entity_news[j])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=True)
        entity_news[j] = np.array([t_entity_news[j][k] for k in sampled_indices])
    news_title = np.array(news_title)

    news, news_title, news_entity, entity_news

    with open(USER_NEWS_FILE, 'r') as file:
        users = json.load(file)
    len_news = len(news)
    t_user_news = defaultdict(list)
    t_news_user = defaultdict(list)
    user_news = np.zeros([1+len(users), args.news_neighbor], dtype=np.int32)
    news_user = np.zeros([1+len_news, args.user_neighbor], dtype=np.int32)
    data = []
    for user in users:
        for i in range(len(users[user]) - 1):
            t_user_news[int(user)].append(users[user][i]['id'])
            t_news_user[users[user][i]['id']].append(int(user))

        # sample news neighbors of user
        n_neighbors = len(t_user_news[int(user)])
        if n_neighbors >= args.news_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.news_neighbor,
                                                replace=True)
        user_news[int(user)] = np.array([t_user_news[int(user)][i] for i in sampled_indices])



        t1 = trans_time(users[user][-1]['time'], users[user][-1]['publishtime'])
        data.append([user, users[user][-1]['id'], t1, 1])

        read_news = [x['id'] for x in users[user]]
        negtive = str(random.sample(set(range(1, len_news + 1)) - set(read_news), 1)[0])
        t2 = trans_time(news[negtive]['time'], news[negtive]['publishtime'])
        data.append([user, negtive, t2, 0])
    # sample user neighbors of news
    for i in range(1,len(t_news_user)+1):
        n_neighbors = len(t_news_user[i])
        if n_neighbors >= args.user_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor,
                                                replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.user_neighbor,
                                                replace=True)
        news_user[i] = np.array([t_news_user[i][j] for j in sampled_indices])

    # dataset split
    train_data, eval_data, test_data = dataset_split(np.array(data), args)

    train_data, eval_data, test_data, user_news, news_user

    news, news_title, news_entity, entity_news = load_news(args)
    train_data, eval_data, test_data, user_news, news_user = load_events(news, args)
    entity_entity = load_entity()


    news_group = entity_news


    # test_data = np.array(test_data)

    eval_indices = np.random.choice(list(range(test_data.shape[0])), size=int(test_data.shape[0] * 0.2), replace=False)
    test_indices = list(set(range(test_data.shape[0])) - set(eval_indices))
    eval_data = test_data[eval_indices]
    test_data = test_data[test_indices]

    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    l = int(len(test_data) * 0.1)

    eval_data = test_data[:l]
    test_data = test_data[l:]


    train_data, eval_data, test_data, train_user_news, train_news_user, test_user_news, test_news_user, news_title, news_entity, news_group


# main()
