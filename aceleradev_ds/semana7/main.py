#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[526]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import (ColumnTransformer, make_column_selector as selector)
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler
)
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[527]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[528]:


countries = pd.read_csv("countries.csv")


# In[529]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[530]:


# Sua análise começa aqui.
countries.info()


# In[531]:


#Frequencia de valores unicos
countries['Climate'].value_counts()


# In[532]:


#Colunas para correcao de tipo
cols_to_float = ["Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"]

#Formas de conversao to_numeric
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html 
#countries[cols_to_float] = countries[cols_to_float].apply(pd.to_numeric)

#astype
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
#countries.Pop_density.str.replace(',','.').astype(float)

countries[cols_to_float] = countries[cols_to_float].apply(lambda s: s.str.replace(',','.').astype(float))

countries[cols_to_float]


# In[533]:


countries.info()


# In[534]:


df_info_nan = pd.DataFrame({'dtypes': countries.dtypes,
                            'total_nan': countries.isna().sum(), 
                            'perc': countries.isna().sum()/len(countries)})
df_info_nan


# In[535]:


#Listagem de linhas com pelo menos x colunas nan
countries[countries.isna().sum(axis=1) > 3]


# In[536]:


#Trim com strip
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.strip.html

cols_to_trim = ["Country", "Region",]
countries[cols_to_trim] = countries[cols_to_trim].apply(lambda s: s.str.strip())


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[537]:


countries.Region.unique()


# In[538]:


def q1():
    regioes = countries.Region.unique()
    regioes.sort()

    return list(regioes)


# In[539]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[540]:


#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

pop_density_bins = discretizer.fit_transform(countries[["Pop_density"]])

pop_density_bins[:5]


# In[541]:


np.unique(pop_density_bins)


# In[542]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int((pop_density_bins >= 9).sum())


# In[543]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[544]:


#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)

region_climate_encoded = one_hot_encoder.fit_transform(countries[["Region", "Climate"]].fillna(0))


region_climate_encoded.shape


# In[545]:


#outra forma para Q3
countries[["Region", "Climate"]].fillna(0).nunique().sum()


# In[546]:


one_hot_encoder.categories_


# In[547]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return region_climate_encoded.shape[1] 


# In[548]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[549]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[550]:



#Copia superficial de countries
ppln_data = countries.copy()

#https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html?highlight=pipeline#sklearn.pipeline.Pipeline
#Passos da pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
])


# In[551]:


#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
#Colunas com Int e Float
cols_num = countries.select_dtypes(include=['int64','float64']).columns
cols_object = countries.select_dtypes(exclude=['int64','float64']).columns

numeric_transformer.fit(countries[cols_num])

#Testes Reshape 
#test_country_r = np.reshape(test_country[2:], (1, -1)) #Para 2D
#test_country_t = numeric_transformer.transform([test_country[2:]])

#test_country_t


# In[552]:


df_test_country = pd.DataFrame([test_country], columns=countries.columns)

df_test_country.head()


# In[553]:


test_country_transformed = numeric_transformer.transform(df_test_country[cols_num])

test_country_transformed


# In[554]:


df_test_country_results = pd.DataFrame(test_country_transformed, columns=cols_num)

df_test_country_results.head()


# In[555]:


#Concatenando os dados object com os resultados
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
df_test_country_results = pd.concat([df_test_country[cols_object], df_test_country_results], axis=1)

df_test_country_results.head()


# In[556]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return float(round(df_test_country_results.Arable, 3))


# In[557]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[558]:


net_migration_outliers = countries.Net_migration.copy()


# In[559]:


figsize(12,8)
sns.boxplot(net_migration_outliers, orient="vertical")


# In[560]:


#primeiro, terceiro quartis e intervalo interquartil
quartil1, quartil3 = net_migration_outliers.quantile([.25, .75])
iiquartil = quartil3 - quartil1

iiquartil


# $[Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}]$

# In[561]:


non_outlier_interval_iqr = [quartil1 - 1.5 * iiquartil, quartil3 + 1.5 * iiquartil]

non_outlier_interval_iqr


# In[562]:


below_outliers_iqr = net_migration_outliers[net_migration_outliers < non_outlier_interval_iqr[0]]
above_outliers_iqr = net_migration_outliers[net_migration_outliers > non_outlier_interval_iqr[1]]

print(f"Total de observacoes: {len(net_migration_outliers)} " +
        f"\nTotal de outliers: {len(below_outliers_iqr) + len(above_outliers_iqr)} " +
        f"\nPercentual de outliers: {(len(below_outliers_iqr) + len(above_outliers_iqr))/len(net_migration_outliers)} " +
        f"\nPrimeiros 5 outiliers abaixo do intervalo: \n{below_outliers_iqr.head()}" +
        f"\nPrimeiros 5 outiliers acima do intervalo: \n{above_outliers_iqr.head()}")


# In[563]:


def q5():
    # Retorne aqui o resultado da questão 4.
    #Como temos 22% de outliers, precisaria de melhor analise para decidir como tratar.
    is_removed = False
    return (len(below_outliers_iqr), len(above_outliers_iqr), is_removed)


# In[564]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[565]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[566]:


#Temos um corpus com 1773 documentos
len(newsgroups.data)


# In[567]:


#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)


# In[568]:


#Indice da palavra phone
word_phone_idx = count_vectorizer.vocabulary_.get("phone")

#Ocorrencias em cada doc da palavra phone
word_phone_in_docs = newsgroups_counts[:-1,word_phone_idx].toarray()


# In[569]:


def q6():
    # Retorne aqui o resultado da questão 4.
    return int(word_phone_in_docs.sum())


# In[570]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[571]:


#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html?highlight=tfidfvectorizer#sklearn.feature_extraction.text.TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(newsgroups.data)

newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups.data)


# In[572]:


newsgroups_tfidf_vectorized.shape


# In[573]:


phone_tfidf = newsgroups_tfidf_vectorized[:, word_phone_idx].toarray()
phone_tfidf[phone_tfidf > 0]


# In[574]:


pd.DataFrame(phone_tfidf, columns=[(tfidf_vectorizer.get_feature_names())[word_phone_idx]])


# In[575]:


def q7():
    # Retorne aqui o resultado da questão 4.
    return float(round(phone_tfidf.sum(),3))


# In[576]:


q7()

