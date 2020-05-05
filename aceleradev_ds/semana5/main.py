#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm #para qq


# In[3]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
athletes.head()


# In[7]:


athletes.describe()


# In[8]:


athletes.info()


# In[9]:


athletes.isna().sum()


# In[10]:


athletes.shape


# In[11]:


#Questão 1
amostra_1_height = get_sample(athletes, 'height', n=3000).to_frame()
amostra_1_height.head()


# In[12]:


amostra_1_height.describe()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[13]:


shapiro_amostra_1 = sct.shapiro(amostra_1_height) #Returns W: float (the test statistic), p-value: float, aplha = 0.05
print(shapiro_amostra_1)
shapiro_amostra_1[1] > 0.05


# In[14]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return bool(shapiro_amostra_1[1] >= 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[99]:


sns.distplot(amostra_1_height, bins=25)
plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.title("Amostra 1: Altura")
plt.show()


# In[111]:


sm.qqplot(data=amostra_1_height.height, fit=True, line="45");


# In[17]:


shapiro_amostra_1[1] > 0.0000001


# In[18]:


#Questão 2
jarque_bera_amostra1 = sct.jarque_bera(amostra_1_height)
print(jarque_bera_amostra1)
jarque_bera_amostra1[1] > 0.05


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[19]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return bool(jarque_bera_amostra1[1] >= 0.05)


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[20]:


amostra_2_weight = get_sample(athletes, 'weight', n=3000).to_frame()
amostra_2_weight.head()


# In[21]:


amostra_2_weight.describe()


# In[22]:


agostino_pearson_amostra2 = sct.normaltest(amostra_2_weight)
print(agostino_pearson_amostra2)
agostino_pearson_amostra2[1][0] > 0.05


# In[23]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return bool(agostino_pearson_amostra2[1][0] >= 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[24]:


sns.distplot(amostra_2_weight, bins=25);
plt.xlabel("Peso")
plt.ylabel("Frequência")
plt.title("Amostra 2: Peso")
plt.show()


# In[25]:


sns.boxplot(data=amostra_2_weight)
plt.xlabel("Peso")
plt.ylabel("Valores")
plt.title("Amostra 2: Peso")
plt.show()


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[26]:


#Questão 4
amostra_2_weight_transformada = np.log10(amostra_2_weight)
amostra_2_weight_transformada


# In[27]:


agostino_pearson_amostra2_transformada = sct.normaltest(amostra_2_weight_transformada)
print(agostino_pearson_amostra2_transformada)
agostino_pearson_amostra2_transformada[1][0] > 0.05


# In[28]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return bool(agostino_pearson_amostra2_transformada[1][0] >= 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[29]:


sns.distplot(amostra_2_weight_transformada, bins=25);
plt.xlabel("Peso log")
plt.ylabel("Frequência")
plt.title("Amostra 2: Peso log")
plt.show()


# In[143]:


sns.boxplot(data=amostra_2_weight_transformada)
plt.xlabel("Peso log")
plt.ylabel("Valores")
plt.title("Amostra 2: Peso log")
plt.show()


# In[109]:


sm.qqplot(amostra_2_weight_transformada.weight, fit=True, line="45");


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[31]:


bra = athletes[athletes.nationality == 'BRA']
usa = athletes[athletes.nationality == 'USA']
can = athletes[athletes.nationality == 'CAN']


# In[32]:


bra.head()


# In[33]:


bra.describe()


# In[34]:


bra.isna().sum()


# In[35]:



bra = bra.dropna(subset=['height'])
bra.isna().sum()


# In[36]:


usa.head()


# In[37]:


usa.describe()


# In[38]:


usa.isna().sum()


# In[39]:


usa = usa.dropna(subset=['height'])
usa.isna().sum()


# In[40]:


can.head()


# In[41]:


can.describe()


# In[42]:


can.isna().sum()


# In[43]:


can = can.dropna(subset=['height'])
can.isna().sum()


# In[50]:


#equal_var = False para variâncias diferentes
bra_usa_ttest_height = sct.ttest_ind(bra['height'], usa['height'], equal_var=False)
bra_usa_ttest_height


# In[45]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return bool(bra_usa_ttest_height.pvalue >= 0.05)


# In[122]:


sns.distplot(bra['height'], bins=25, hist=False, rug=True, label='BRA')
sns.distplot(usa['height'], bins=25, hist=False, rug=True, label='USA')

plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.title("Alturas: BRA x USA")
plt.show()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[53]:


bra_can_ttest_height = sct.ttest_ind(bra['height'], can['height'], equal_var=False)
bra_can_ttest_height


# In[47]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return bool(bra_can_ttest_height.pvalue >= 0.05)


# In[119]:


sns.distplot(bra['height'], bins=25, hist=False, rug=True, label='BRA')
sns.distplot(can['height'], bins=25, hist=False, rug=True, label='CAN')

plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.title("Alturas: BRA x CAN")
plt.show()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[114]:


usa_can_ttest_height = sct.ttest_ind(usa['height'], can['height'], equal_var=False)
print(usa_can_ttest_height)
print(float(np.round(usa_can_ttest_height.pvalue, 8)))


# In[113]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(np.round(usa_can_ttest_height.pvalue, 8))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[96]:


#Testes para entender as informações das funções

usa_can_ttest_height_eq_var = sct.ttest_ind(usa['height'], can['height'], equal_var=True)
print(usa_can_ttest_height_eq_var)
#sct.t.isf(.05 / 2, gl) 
#sct.t.isf(0.00046601/2, gl)
#sct.t.sf(1.9626695531494305, gl, loc=0, scale=1) *2   
#sct.t.sf(3.516987632488539, gl, loc=0, scale=1)*2
#sct.t.pdf(usa_can_ttest_height.statistic, 13)

#grau de liberdade para o teste t independente com variancias semelhantes: df = n1 + n2 - 2
gl = len(usa) + len(can) - 2
print(f"Graus de liberdade: {gl}")
q7_sf = sct.t.sf(usa_can_ttest_height_eq_var.statistic, gl)*2 #Para Hipótese Bicaudal
print(q7_sf)


# In[118]:


sns.distplot(usa['height'], bins=25, hist=False, rug=True, label='USA')
sns.distplot(can['height'], bins=25, hist=False, rug=True, label='CAN')

plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.title("Alturas: USA x CAN")
plt.show()


# In[125]:



sns.distplot(usa['height'], bins=25, hist=False, rug=True, label='USA')
sns.distplot(can['height'], bins=25, hist=False, rug=True, label='CAN')
sns.distplot(bra['height'], bins=25, hist=False, rug=True, label='BRA')

plt.xlabel("Altura")
plt.ylabel("Frequência")
plt.title("Alturas: USA x CAN x BRA")
plt.show()

