#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import mlflow
import mlflow.sklearn

import gc  # Consumo de memória é muito alto, então chamada
            # ao coletor de lixo ajuda um pouco
import os
import json
import time
import pickle
import importlib
import itertools

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint as sp_randint

import sys
sys.path.append('..')

import util
import upbg


# In[2]:


T0 = time.time()

def sec_exp(t): 
    d = t // 86400; t -= d * 86400 
    h = t // 3600; t -= h * 3600 
    m = t // 60; t -= m * 60 
    s = t; t -= s 
    return dict(d=d, h=h, m=m, s=s) 

def beautiful_date(dic):
    return "{}d, {}h, {}m e {:.2f}s".format(
        dic.get('d', 0), dic.get('h', 0), dic.get('m'), dic.get('s')
    )

def show_spent_time(t, s='', t0=T0):
    spent_time = beautiful_date(sec_exp(t - t0))
    print(s or "Tempo desde o começo do experimento:", spent_time)


# In[3]:

RUN = mlflow.start_run()
RUN_ID = mlflow.active_run().info.run_id
print("RUN_ID:", RUN_ID)


parser = argparse.ArgumentParser(description='[Main] Run PBG on jurix2020 corpus.')
parser.add_argument('--n_components', type=int, default=50)
parser.add_argument('--alpha', type=float, default=0.005)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--local_max_itr', type=int, default=20)
parser.add_argument('--global_max_itr', type=int, default=50)
parser.add_argument('--local_threshold', type=float, default=1e-6)
parser.add_argument('--global_threshold', type=float, default=1e-6)
parser.add_argument('--ngram_min', type=int, default=1)
parser.add_argument('--ngram_max', type=int, default=1)
parser.add_argument('--lines_percentage', type=float, default=1, 
    help='How many lines will be used for train/test/validation datasets',
)
parser.add_argument('--data_size', type=str, default='small',
    choices=['small', 'medium'],
    help='Which dataset will be used for train/test/validation ',
)
parser.add_argument('--use_cache', type=int, default=1,
    choices=[1, 0],
    help='Whether to use the existing previous preprocessed train dataset.',
)
parser.add_argument('--log_model', type=int, default=0)



args, unknown = parser.parse_known_args()
dict_args = args.__dict__

# mlflow.log_params(dict_args)

json.dump(dict_args, open('params.json', 'w'), indent=4*' ', ensure_ascii=False)
mlflow.log_artifact("params.json")
print("ARGS: ", args)
open('log.txt', 'w').write(str(args))


# In[4]:


DATA_SIZE=dict_args.pop('data_size', 'small')
VALIDATION_DATA_PATH='csv/validation_{}.csv'.format(DATA_SIZE)
TRAIN_DATA_PATH='csv/train_{}.csv'.format(DATA_SIZE)
TEST_DATA_PATH='csv/test_{}.csv'.format(DATA_SIZE)

THEMES = [
    5, 6, 26, 33, 139, 163, 232, 313, 339, 350, 406, 409,
    555, 589, 597, 634, 660, 695, 729, 766, 773, 793, 800,
    810, 852, 895, 951, 975
]


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
    data = groupby_process(data)
    
    data.themes = data.themes.apply(lambda x: eval(x))
    return data

def transform_y(train_labels, test_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)

    mlb_train = mlb.transform(train_labels)
    mlb_test = mlb.transform(test_labels)

    print(mlb.classes_)

    return mlb_train, mlb_test, mlb


# In[5]:

LINES_PERCENTAGE = dict_args.pop('lines_percentage')
train_data = get_data(TRAIN_DATA_PATH, lines_per=LINES_PERCENTAGE)
test_data = get_data(TEST_DATA_PATH, lines_per=LINES_PERCENTAGE)

print("MAX_TRAIN_TYPE:", type(train_data.body[0]))
print("MAX_TRAIN_LEN:", max(map(len, train_data.body)))
print("MAX_TEST_LEN:", max(map(len, test_data.body)))



# In[6]:


train_data.themes = train_data.themes.apply(
    lambda x: list(set(sorted([i if i in THEMES else 0 for i in x])))
)
test_data.themes = test_data.themes.apply(
    lambda x: list(set(sorted([i if i in THEMES else 0 for i in x])))
)
# validation_data.themes = validation_data.themes.apply(lambda x: list(set(sorted([i if i in THEMES else 0 for i in x]))))


y_train, y_test, mlb = transform_y(train_data.themes, test_data.themes)
del y_test; gc.collect()
del test_data; gc.collect()

X_train = train_data.body
# X_test = test_data.body
X_train_themes = train_data.themes

print('X_train: {}, \n\ty_train: {}'.format(X_train.shape, y_train.shape))
# print('X_test: {}, \n\ty_test: {}'.format(X_test.shape, y_test.shape))


vectorizer = TfidfVectorizer(
    ngram_range=(dict_args.pop('ngram_min'), dict_args.pop('ngram_max')),
    sublinear_tf=True,
)


X_train_vect = vectorizer.fit_transform(X_train)
# X_valid = vectorizer.transform(validation_data.body)
# X_test_vect = vectorizer.transform(X_test)
# y_valid = mlb.transform(validation_data.themes)

del X_train
# del X_test
gc.collect()


# In[7]:


# %%time
import util
importlib.reload(util)
# DISCLAIMER: só pode ser executada uma vez (não pergunte o motivo,
# mas parece ter a ver com `docs` do SimplePreprocessingBR)

print('preprocessing...')
params=dict(
    use_nltk=True,
    extra_stop_words=[i.lower().strip() for i in open('stopwords.txt').readlines()],
)

# pp = util.SimplePreprocessingBR_Lite(**params)
t0_proc_train = time.time()

preprocessed_path = f'preprocessed/train_{DATA_SIZE}_{LINES_PERCENTAGE:.2f}.pkl'
use_cache = dict_args.pop('use_cache', )
print("USE_CACHE:", str(use_cache))
print("PREPROCESSED_PATH:", preprocessed_path)
if not use_cache:
    pp = util.SimplePreprocessing_MemConstrained(**params)
    M_train = pp.transform(train_data.body)
    print("Dumping proprocessed_train...")
    os.makedirs('preprocessed', exist_ok=True)
    pickle.dump(M_train, open(preprocessed_path, 'wb'))

    del pp; gc.collect()
else:
    print("Using cached preprocessed train...")
    M_train = pickle.load(open(preprocessed_path, 'rb'))

show_spent_time(time.time(), f"Tempo gasto preprocessando treino:", t0_proc_train)

print("PREPROCESSOU O TREINO")
print('done.')

categories = set((itertools.chain(*train_data.themes)))
n_class = len(categories)
print(f'nclass {n_class}')


M_train_vectorized = vectorizer.fit_transform(M_train)
del M_train; gc.collect()


pbg = upbg.UPBG(
    **dict_args,
    feature_names=vectorizer.get_feature_names(),
    debug=True,
)

print("fitting...")
# pbg.fit(M_train, newsgroups_train.target)
t0_fit = time.time()
pbg.fit(
    M_train_vectorized,
    X_train_themes,
)
show_spent_time(time.time(), "Tempo gasto no treinamento:", t0_fit)
print('done')

# dump de modelo opcional. Afinal eles podem ser bem grandes e estourar armazenamento
if dict_args.pop('log_model'):
    print("Gonna LOG the model...")
    mlflow.sklearn.log_model(pbg, 'pbg_model_spacy')


# ## Vai fazer dump tema-topico

temas = json.load(open('temas.json')) 
tema_topico = {} 
topicos = pbg.get_topics(20) 
for tema, topico in pbg.map_class_.items(): 
    if tema > 0: 
        tema = str(tema)
        tema_topico[temas[tema]] = topicos[topico] 
mlflow.log_dict(tema_topico, "tema_topico.json", )





import dump_temas_e_topicos

tema_topico = dump_temas_e_topicos.get_tema_topico(pbg)
topicos_dict_sem_tema = dump_temas_e_topicos.get_topicos_sem_tema(pbg)

DUMP_FOLDER = 'tema_topico'
os.makedirs(DUMP_FOLDER, exist_ok=True)

tema_topico_path = f'{DUMP_FOLDER}/tema_topico_{RUN_ID}.json'
json.dump(
    tema_topico,
    open(tema_topico_path, 'w'),
        indent=4*' ', ensure_ascii=False
) 

topicos_dict_sem_tema_path = f'{DUMP_FOLDER}/topicos_sem_tema_{RUN_ID}.json'
json.dump(
    topicos_dict_sem_tema,
    open(topicos_dict_sem_tema_path, 'w'),
    indent=4*' ', ensure_ascii=False
)

mlflow.log_artifact(tema_topico_path)
mlflow.log_artifact(topicos_dict_sem_tema_path)

# Abaixo, é chegado o momento de fazer dump tema_topico e dos topicos sem temas pois, por algum motivo, acho que o dump dos modelos treinados com a base `medium` estão falhando.

# In[ ]:




