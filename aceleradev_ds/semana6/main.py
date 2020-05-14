#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from loguru import logger


# In[4]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


fifa = pd.read_csv("fifa.csv")


# In[6]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.
fifa.head()


# In[8]:


fifa.info()


# In[9]:


fifa.describe()


# In[10]:


fifa.isna().sum()


# In[11]:


fifa[fifa.isna().sum(axis=1) > 0].head(10)


# In[12]:


fifa = fifa.dropna()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[13]:


pca = PCA().fit(fifa)

explained_var_ratio = pca.explained_variance_ratio_

explained_var_ratio


# In[14]:


def q1():
    # Retorne aqui o resultado da questão 1.
    first_pca_explained_var_ratio  = explained_var_ratio[0]
    return float(round(first_pca_explained_var_ratio, 3))


# In[15]:


q1()


# In[16]:


per_explained_var_ratio = np.round(explained_var_ratio * 100, decimals=1)

#labels = ['PC' + str(i) for i in range(1, len(per_explained_var_ratio) + 1)]

plt.figure(figsize=(16,10))

plt.bar(x=range(1, len(per_explained_var_ratio) + 1), height=per_explained_var_ratio)
plt.show()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[17]:


cumulative_var_ratio = np.cumsum(explained_var_ratio)
component_number = np.argmax(cumulative_var_ratio >= 0.95) + 1 # Contagem começa em zero.

plt.plot(cumulative_var_ratio)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[18]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int(component_number)


# In[19]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[20]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[21]:


pca.components_


# In[22]:


x_pca = np.dot(pca.components_[0:2], x)

x_pca


# In[23]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return (round(x_pca[0], 3), round(x_pca[1], 3))


# In[24]:


q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# 
# Feature Selection: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# 
# 
# 

# In[27]:


y = fifa['Overall']
fifa_without_overall = fifa.drop(['Overall'], axis=1)

model = LinearRegression()

#Initializing RFE model
rfe = RFE(model, n_features_to_select=5)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(fifa_without_overall,y)  

print(X_rfe)

selected_features_rfe = fifa_without_overall.columns[rfe.support_]
print(selected_features_rfe)


# In[152]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return list(selected_features_rfe)


# In[153]:


q4()


# Example of number of features where the accuracy is the highest.

# In[154]:


#Correlation check
plt.figure(figsize=(16,12))
fifa_corr = fifa.corr()
sns.heatmap(fifa_corr, cmap=plt.cm.Reds)
plt.show()


# In[155]:


fifa_corr


# ```Python
# #no of features
# nof_list=np.arange(1,fifa_without_overall.shape[1])            
# high_score=0
# #Variable to store the optimum features
# nof=0           
# score_list =[]
# for n in range(len(nof_list)):
#     X_train, X_test, y_train, y_test = train_test_split(fifa_without_overall,y, test_size = 0.3, random_state = 0)
#     model = LinearRegression()
#     rfe = RFE(model,nof_list[n])
#     X_train_rfe = rfe.fit_transform(X_train,y_train)
#     X_test_rfe = rfe.transform(X_test)
#     model.fit(X_train_rfe,y_train)
#     score = model.score(X_test_rfe,y_test)
#     score_list.append(score)
#     if(score>high_score):
#         high_score = score
#         nof = nof_list[n]
# print("Optimum number of features: %d" %nof)
# print("Score with %d features: %f" % (nof, high_score))
# ```
