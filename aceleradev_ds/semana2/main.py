#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[5]:


black_friday.head()


# In[6]:


black_friday.shape


# In[24]:


n_observacoes = black_friday.shape[0]
n_colunas = black_friday.shape[1]


# In[7]:


black_friday.info()


# In[8]:


black_friday['Age'].unique()


# In[23]:


black_friday.loc[black_friday['Age'] == '26-35'].head()


# In[26]:


black_friday.loc[black_friday['Age'] == '26-35'].shape


# In[29]:


#Genero Feminino com idade entre 26 e 35
black_friday.loc[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')].head()


# In[44]:


black_friday.nunique()


# In[43]:


black_friday['User_ID'].nunique()


# In[46]:


black_friday.describe()


# In[49]:


black_friday.dtypes


# In[18]:


black_friday.dtypes.count()


# In[20]:


tipos_colunas = black_friday.dtypes.unique().shape[0]
tipos_colunas


# In[55]:


black_friday.isna().sum()


# In[22]:


registros_nulos = black_friday.isna().sum(axis=1).loc[lambda x : x>0].count()


# In[25]:


porcentagem_registros_nulos = float(registros_nulos/n_observacoes)
porcentagem_registros_nulos


# In[59]:


#Numero de nulos por linha
black_friday.isna().sum(axis=1)


# In[61]:


black_friday.isna().sum()


# In[62]:


black_friday.isna().sum().max()


# In[101]:


black_friday['Product_Category_3'].mode()


# In[40]:


black_friday_purchase = black_friday['Purchase']
black_friday_purchase


# In[41]:


media_purchase = black_friday_purchase.mean()
media_purchase


# In[42]:


desvio_padrao_purchase = black_friday_purchase.std()
desvio_padrao_purchase


# In[45]:


black_friday_purchase_padronizado = (black_friday_purchase - media_purchase) / desvio_padrao_purchase
black_friday_purchase_padronizado


# In[53]:


black_friday_purchase_padronizado.loc[lambda x: (x >= -1) & (x <= 1)]


# In[54]:


black_friday_purchase_padronizado.loc[lambda x: (x >= -1) & (x <= 1)].count()


# In[55]:


black_friday_purchase_normal = (black_friday_purchase - black_friday_purchase.min()) / (black_friday_purchase.max() - black_friday_purchase.min())
black_friday_purchase_normal


# In[56]:


black_friday_purchase_normal.mean()


# In[4]:


black_friday['Product_Category_2'].isna().sum()


# In[5]:


black_friday['Product_Category_3'].isna().sum()


# In[11]:


black_friday['Product_Category_2'].isna().loc[lambda x: x == True]


# In[10]:


black_friday['Product_Category_3'].isna().loc[lambda x: x == True]


# In[19]:


product_category_2_index = black_friday['Product_Category_2'].isna().loc[lambda x: x == True].index
product_category_2_index


# In[18]:


product_category_3_index = black_friday['Product_Category_3'].isna().loc[lambda x: x == True].index
product_category_3_index


# In[22]:


product_category_2_em_category3 = product_category_2_index.isin(product_category_3_index)
product_category_2_em_category3


# In[26]:


total_product_category_2_em_category3 = product_category_2_em_category3.sum()
total_product_category_2_em_category3


# In[28]:


#Se o total de nulos (true) de Product_Category_2 que estão em Product_category_3 é igual ao total de nulos de Category 2
#entao todos os nulos de 2 estão em 3
is_total_product_category_2_em_category3 = total_product_category_2_em_category3 == black_friday['Product_Category_2'].isna().sum()
is_total_product_category_2_em_category3


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[12]:


def q1():
    # Retorne aqui o resultado da questão 1.
    n_observacoes = black_friday.shape[0]
    n_colunas = black_friday.shape[1]
    return (n_observacoes, n_colunas)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[30]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday.loc[(black_friday['Age'] == '26-35') & (black_friday['Gender'] == 'F')].shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[45]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[83]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return tipos_colunas


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[88]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return porcentagem_registros_nulos


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[63]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[107]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(black_friday['Product_Category_3'].mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[57]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return float(black_friday_purchase_normal.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[58]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return int(black_friday_purchase_padronizado.loc[lambda x: (x >= -1) & (x <= 1)].count())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[29]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return bool(is_total_product_category_2_em_category3)

