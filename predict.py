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

## --- leitura dos dados de teste

test_data = reduce_mem_usage(pd.read_csv('./inputs/desafio_de_credito_test.csv', engine='c'))

## --- ajustes dos tipos de dados

test_data['var_f']    = test_data['var_f'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)
test_data['borrowed'] = test_data['borrowed'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)
test_data['income']   = test_data['income'].apply(lambda x: tryconvert(x, np.NaN, float)).astype(float)

test_data['reason']     = test_data['reason'].astype(str)
test_data['state']      = test_data['state'].astype(str)
test_data['zipcode']    = test_data['zipcode'].astype(str)
test_data['channel']    = test_data['channel'].astype(str)
test_data['job_name']   = test_data['job_name'].astype(str)
test_data['real_state'] = test_data['real_state'].astype(str)

test_data['social_network'] = test_data['social_network'].apply(lambda x: 'True' if str(x).lower()[0] == 't' else
                                                                         'False' if str(x).lower()[0] == 'f' else x)

## --- imputação de dados

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


## --- eliminação das variáveis menos relevantes
test_data.drop('channel', axis=1, inplace=True)
test_data.drop(['n_defaulted_loans','n_bankruptcies'], axis=1, inplace=True)

## --- preenchimento de valores faltantes
## --- preenchimento com a moda
test_data['sex'].fillna(test_data['sex'].mode()[0], inplace=True)

## --- preenchimento com a media
test_data['n_issues'].fillna( test_data[~test_data['n_issues'].isna()]['n_issues'].astype(int).mean(), inplace=True )

## --- preenchimento com valor arbitrário
test_data['sign']           = test_data['sign'].apply(lambda x: 'ARB' if type(x) == pd._libs.missing.NAType else x).astype(str)
test_data['social_network'] = test_data['social_network'].apply(lambda x: 'ARB' if type(x) == float else x).astype(str)
test_data['ok_since']       = test_data['ok_since'].apply(lambda x: -3290 if (type(x) == pd._libs.missing.NAType) else
                                      -3290 if (str(x)[0].isnumeric() == False) else x).astype(float)

## -- encoding das variáveis
for col in test_data.columns:
    if (test_data[col].dtype == object) | (test_data[col].dtype == "string[pyarrow]"):
        enc             = LabelEncoder()
        enc.classes_    = np.load('./encoders/classes_%s.npy' % col, allow_pickle=True)
        test_data[col]  = enc.transform(test_data[col])


clf = pickle.load(open('./models/classificador', 'rb'))
        
## --- classificações
test_data['default'] = clf.predict(test_data)
test_data['default'] = test_data['default'].astype(bool)
test_data['probas']  =  [x[0] if x[0] > x[1] else x[1] for x in clf.predict_proba(test_data.drop('default', axis=1))]

## --- gravação do arquivo final
test_data.to_csv(open('./outputs/final_submissions.csv', 'w'), index=False)

desafio_test = reduce_mem_usage(pd.read_csv('./inputs/desafio_de_credito_test.csv', engine='c'))

desafio_test['default'] = test_data['default']
desafio_test['probas']  = test_data['probas']

desafio_test.to_csv(open('./outputs/desafio_de_credito_test.csv', 'w'), index=False)
