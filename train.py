## --- bibliotecas de estruturas de dados e visualização
import pandas as pd
import numpy as np

## --- funções de pré-processamento
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

## --- bibliotecas de machine learning
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

import pickle

from subroutines import reduce_mem_usage, tryconvert
import warnings

warnings.filterwarnings('ignore')

## --- leitura dos dados de treino

df = reduce_mem_usage(pd.read_csv('./inputs/desafio_de_credito_train.csv', engine='c'))
test_data = reduce_mem_usage(pd.read_csv('./inputs/desafio_de_credito_test.csv', engine='c'))


## --- ajustes dos tipos de dados
df['var_f']    = df['var_f'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)
df['borrowed'] = df['borrowed'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)
df['income']   = df['income'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)

test_data['var_f']    = test_data['var_f'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)
test_data['borrowed'] = test_data['borrowed'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)
test_data['income']   = test_data['income'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)

df['reason']     = df['reason'].astype(str)
df['state']      = df['state'].astype(str)
df['zipcode']    = df['zipcode'].astype(str)
df['channel']    = df['channel'].astype(str)
df['job_name']   = df['job_name'].astype(str)
df['real_state'] = df['real_state'].astype(str)

test_data['reason']     = test_data['reason'].astype(str)
test_data['state']      = test_data['state'].astype(str)
test_data['zipcode']    = test_data['zipcode'].astype(str)
test_data['channel']    = test_data['channel'].astype(str)
test_data['job_name']   = test_data['job_name'].astype(str)
test_data['real_state'] = test_data['real_state'].astype(str)

df['social_network'] = df['social_network'].apply(lambda x: 'True' if str(x).lower()[0]  == 't' else
                                                            'False' if str(x).lower()[0] == 'f' else x)

test_data['social_network'] = test_data['social_network'].apply(lambda x: 'True' if str(x).lower()[0] == 't' else
                                                                         'False' if str(x).lower()[0] == 'f' else x)


## --- imputação de dados
df.drop(['id','Unnamed: 27'], axis=1, inplace=True)
df.dropna(axis=0, how='all',inplace=True )
df.drop(df[df['sign']=='73007.68'].index, axis=0, inplace=True)
df['var_f']    = df['var_f'].astype(float)
df['income']   = df['income'].astype(float)
df['borrowed'] = df['borrowed'].astype(float)
df['months'].fillna(df['months'].mode()[0], inplace=True)
df['months']   = df['months'].astype('int16')
df['reason'].fillna(df['reason'].mode()[0], inplace=True)
df['income'].fillna(df['income'].median(), inplace=True)
df.loc[df[df['limit'].isna()].index, 'limit'] = -3290


test_data.drop(['id','Unnamed: 27'], axis=1, inplace=True)
test_data.dropna(axis=0, how='all',inplace=True )
test_data['var_f']    = test_data['var_f'].astype(float)
test_data['income']   = test_data['income'].astype(float)
test_data['borrowed'] = test_data['borrowed'].astype(float)
test_data['months'].fillna(test_data['months'].mode()[0], inplace=True)
test_data['months']   = test_data['months'].astype('int16')
test_data['reason'].fillna(test_data['reason'].mode()[0], inplace=True)
test_data['income'].fillna(test_data['income'].median(), inplace=True)
test_data.loc[test_data[test_data['limit'].isna()].index, 'limit'] = -3290

## --- separação das targets NaN
def_nan = df[df['default'].isna()]
df.dropna(axis=0, subset=['default'], inplace=True)

df.drop(df['var_f'].sort_values(ascending=False)[:3].index, inplace=True)

## --- eliminação das variáveis menos relevantes
df.drop('channel', axis=1, inplace=True)
def_nan.drop('channel', axis=1, inplace=True)
test_data.drop('channel', axis=1, inplace=True)

df.drop(['n_defaulted_loans','n_bankruptcies'], axis=1, inplace=True)
def_nan.drop(['n_defaulted_loans','n_bankruptcies','default'], axis=1, inplace=True)
test_data.drop(['n_defaulted_loans','n_bankruptcies'], axis=1, inplace=True)
def_nan.dropna(axis=0, how='any', inplace=True)


## --- preenchimento de valores faltantes
## --- preenchimento com a moda
df['sex'].fillna(df['sex'].mode()[0], inplace=True)
def_nan['sex'].fillna(def_nan['sex'].mode()[0], inplace=True)
test_data['sex'].fillna(test_data['sex'].mode()[0], inplace=True)


## --- preenchimento com a media
df['n_issues'].fillna( df[~df['n_issues'].isna()]['n_issues'].astype(int).mean(), inplace=True )
def_nan['n_issues'].fillna( def_nan[~def_nan['n_issues'].isna()]['n_issues'].astype(int).mean(), inplace=True )
test_data['n_issues'].fillna( test_data[~test_data['n_issues'].isna()]['n_issues'].astype(int).mean(), inplace=True )


## --- preenchimento com valor arbitrário
df['sign']           = df['sign'].apply(lambda x: 'ARB' if type(x) == pd._libs.missing.NAType else x).astype(str)
df['social_network'] = df['social_network'].apply(lambda x: 'ARB' if type(x) == float else x).astype(str)
df['ok_since']       = df['ok_since'].apply(lambda x: -3290 if type(x) == pd._libs.missing.NAType else x).astype(float)

def_nan['sign']           = def_nan['sign'].apply(lambda x: 'ARB' if type(x) == pd._libs.missing.NAType else x).astype(str)
def_nan['social_network'] = def_nan['social_network'].apply(lambda x: 'ARB' if type(x) == float else x).astype(str)
def_nan['ok_since']       = def_nan['ok_since'].apply(lambda x: -3290 if type(x) == pd._libs.missing.NAType else x).astype(float)

test_data['sign']           = test_data['sign'].apply(lambda x: 'ARB' if type(x) == pd._libs.missing.NAType else x).astype(str)
test_data['social_network'] = test_data['social_network'].apply(lambda x: 'ARB' if type(x) == float else x).astype(str)
test_data['ok_since']       = test_data['ok_since'].apply(lambda x: -3290 if (type(x) == pd._libs.missing.NAType) else
                                      -3290 if (str(x)[0].isnumeric() == False) else x).astype(float)


composed = pd.concat([df.drop('default', axis=1),
                      test_data[df.drop('default', axis=1).columns]])

composed = composed = pd.concat([composed,
                      def_nan[df.drop('default', axis=1).columns]])

## -- encoding das variáveis
for col in composed.columns:
    if (composed[col].dtype == object) | (composed[col].dtype == "string[pyarrow]"):
        enc         = LabelEncoder()
        fit_enc     = enc.fit(composed[col])
        df[col]     = enc.transform(df[col])
        def_nan[col] = enc.transform(def_nan[col])
        np.save('./encoders/classes_%s.npy' % col, enc.classes_)
        
## --- instanciamento do agrupador
nclust = 2
model  = make_pipeline(StandardScaler(), KMeans(n_clusters   = nclust, 
                                                random_state = 42, 
                                                n_init       = 50, 
                                                max_iter     = 2000, 
                                                tol          = 1e-5))
model.fit(def_nan)

## --- atribui a cada amostra um grupo e salva o agrupador no disco
def_nan['default'] = model.predict(def_nan)
pickle.dump(model, open('./models/clusterizador', 'wb'))


# --- treino do classificador e gravação do modelo no disco

composed = pd.concat([df,def_nan[df.columns]])

X = composed.drop(['default'], axis=1).copy()
y = composed['default'].astype(int)

clf = make_pipeline( StandardScaler(), XGBClassifier(eval_metric       = 'error',
                                                     booster           = 'gbtree',
                                                     learning_rate     = 1.25,
                                                     max_depth         = 5,
                                                     use_label_encoder = False,
                                                     n_estimators      = 90  ) )
clf. fit(X, y)
pickle.dump(clf, open('./models/classificador', 'wb'))