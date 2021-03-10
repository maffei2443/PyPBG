#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('..')


# In[3]:


import mlflow
import mlflow.sklearn

import os
import json
import pickle
import importlib
import itertools
import argparse

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint as sp_randint

from util import SimplePreprocessingBR
import upbg

VALIDATION_DATA_PATH='csv/validation_small.csv'
TRAIN_DATA_PATH='csv/train_small.csv'
TEST_DATA_PATH='csv/test_small.csv'

THEMES = [
    5, 6, 26, 33, 139, 163, 232, 313, 339, 350, 406, 409,
    555, 589, 597, 634, 660, 695, 729, 766, 773, 793, 800,
    810, 852, 895, 951, 975
]
LINES_PERCENTAGE = .05



def groupby_process(df):
    new_df = df.sort_values(['process_id', 'page'])
    new_df = new_df.groupby(
                ['process_id', 'themes'],
                group_keys=False
            ).apply(lambda x: x.body.str.cat(sep=' ')).reset_index()
    new_df = new_df.rename(index=str, columns={0: "body"})
    return new_df

# Nota: para rápida iteração, limitar qtd de linhas carregadas
def get_data(path, preds=None, key=None, lines_per=.02):
    data = pd.read_csv(path)
    if lines_per is not None:
        lines = int(lines_per * data.shape[0])
        data = data.iloc[:lines, :]
    
    data = data.rename(columns={ 'pages': 'page'})
#     data["preds"] = preds[key]
#     data = data[data["preds"] != "outros"]
    data = groupby_process(data)
    
#     data.themes = data.themes.apply(lambda x: literal_eval(x))
    data.themes = data.themes.apply(lambda x: eval(x))
    return data

def transform_y(train_labels, test_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)

    mlb_train = mlb.transform(train_labels)
    mlb_test = mlb.transform(test_labels)

    print(mlb.classes_)

    return mlb_train, mlb_test, mlb


# In[ ]:


parser = argparse.ArgumentParser(description='Run PBG on jurix2020 corpus.')
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--alpha', type=float, default=0.005)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--local_max_itr', type=int, default=1)
parser.add_argument('--global_max_itr', type=int, default=1)
parser.add_argument('--local_threshold', type=float, default=1)
parser.add_argument('--global_threshold', type=float, default=1)
parser.add_argument('--ngram_min', type=int, default=1)
parser.add_argument('--ngram_max', type=int, default=1)
parser.add_argument('--lines_percentage', type=float, default=.1,
    help='How many lines will be used for train/test/validation datasets',
)
args, unknown = parser.parse_known_args()
dict_args = args.__dict__
print("ARGS: ", args)
open('log.txt', 'w').write(str(args))


# In[2]:


train_data = get_data(TRAIN_DATA_PATH, lines_per=dict_args['lines_percentage'])
test_data = get_data(TEST_DATA_PATH, lines_per=dict_args['lines_percentage'])

args.__dict__.pop('lines_percentage', None)
# validation_data = get_data(VALIDATION_DATA_PATH)


# In[3]:


train_data.themes = train_data.themes.apply(
    lambda x: list(set(sorted([i if i in THEMES else 0 for i in x])))
)
test_data.themes = test_data.themes.apply(
    lambda x: list(set(sorted([i if i in THEMES else 0 for i in x])))
)
# validation_data.themes = validation_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))


y_train, y_test, mlb = transform_y(train_data.themes, test_data.themes)

X_train = train_data.body
X_test = test_data.body
print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
# print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))


vectorizer = TfidfVectorizer(
    ngram_range=(dict_args.pop('ngram_min'), dict_args.pop('ngram_max')),
    sublinear_tf=True,
)


X_train_vect = vectorizer.fit_transform(X_train)
# X_valid = vectorizer.transform(validation_data.body)
X_test_vect = vectorizer.transform(X_test)
# y_valid = mlb.transform(validation_data.themes)


# In[9]:


import util
importlib.reload(util)
# DISCLAIMER: só pode ser executada uma vez (não pergunte o motivo,
# mas parece ter a ver com `docs` do SimplePreprocessingBR)

print('preprocessing...')
params=dict(
    use_nltk=True,
    extra_stop_words=[i.lower().strip() for i in open('stopwords.txt').readlines()],
    
)

pp = util.SimplePreprocessingBR(**params)

M_train = pp.transform(train_data.body)
M_test = pp.transform(test_data.body)

print('done.')


# In[10]:


import gc
del pp
gc.collect()


# In[11]:


M_train_vectorized = vectorizer.fit_transform(M_train)
M_test_vectorized = vectorizer.transform(M_test)

# train_data.to_csv("csv/train_ready.csv")
# test_data.to_csv("csv/test_ready.csv")
# validation_data.to_csv("csv/validation_ready.csv")

categories = set((itertools.chain(*train_data.themes)))
n_class = len(categories)

print(f'nclass {n_class}')
# K = 30


labels_raw = [tuple(i) for i in train_data.themes]
mapa = dict([i[::-1] for i in enumerate(set(itertools.chain(*labels_raw)))])
labels_mp = [tuple([mapa[jj] for jj in i]) for i in labels_raw]


# In[ ]:



importlib.reload(upbg)

# hyperparams = dict(
#     n_components=K,
#     alpha=0.005,
#     beta=0.001,
#     local_max_itr=50,
#     global_max_itr=15,
#     local_threshold=1e-6,
#     global_threshold=1e-6,
# )

hyperparams=dict_args

pbg = upbg.UPBG(
    **hyperparams,
    feature_names=vectorizer.get_feature_names(),
)


# In[12]:


print("fitting...")
# pbg.fit(M_train, newsgroups_train.target)
pbg.fit(
    M_train_vectorized,
    [
        t for (idx, t) in enumerate(train_data.themes)
        if len(train_data.body[idx]) <= 1000000
    ],
)
print('done')

mlflow.sklearn.log_model(pbg, 'pbg_model_spacy')


# In[ ]:


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--')

