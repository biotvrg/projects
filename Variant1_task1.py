#!/usr/bin/env python
# coding: utf-8

# Retention – один из самых важных показателей в компании. Ваша задача – написать функцию, которая будет считать retention игроков (по дням от даты регистрации игрока).

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy
import plotly.express as px
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd,MultiComparison
import datetime


# In[3]:


reg = pd.read_csv('~/shared/problem1-reg_data.csv', sep = ';') # Считываем данные


# In[4]:


auth = pd.read_csv('~/shared/problem1-auth_data.csv', sep = ';') # Считываем данные


# In[6]:


reg['reg_ts'] = pd.to_datetime(reg['reg_ts'], unit = 's') # Изменяем тип данных


# In[7]:


auth['auth_ts'] = pd.to_datetime(auth['auth_ts'], unit = 's') # Изменяем тип данных


# In[9]:


all_df = auth.merge(reg, on='uid') # Объединяем два датафрейма


# In[11]:


all_df = all_df[['uid', 'reg_ts', 'auth_ts']] # Меняем местами


# In[12]:


all_df = all_df.rename(columns={'uid' : 'id', 'reg_ts' : 'reg_day', 'auth_ts' : 'auth_day'}) # Переименовываем колонки


# In[14]:


all_df['reg_day'] = all_df['reg_day'].dt.strftime('%m/%d/%Y') # Убираем время из даты


# In[15]:


all_df['reg_day'] = pd.to_datetime(all_df['reg_day']) # Возвращаем необходимый тип данных


# In[16]:


all_df['auth_day']= all_df['auth_day'].dt.strftime('%m/%d/%Y') # Убираем время из даты


# In[17]:


all_df['auth_day'] = pd.to_datetime(all_df['auth_day']) # Возвращаем необходимый тип данных


# In[19]:


all_df = all_df.query('"2020-09-01" <= reg_day and reg_day <= "2020-09-23"') # Выбираем диапазон для расчета метрики


# In[61]:


all_df['days_distance'] = (all_df['auth_day'] - all_df['reg_day']).dt.days + 1 
# Считаем кол-во дней между регистрацией и заходом в игру


# In[62]:


all_df.head()


# In[64]:


ret = all_df.groupby(['reg_day', 'days_distance'])
cohort = ret['id'].size()
cohort = cohort.reset_index()
# Создаем сводную таблицу


# In[65]:


cohort_counts = cohort.pivot(index='reg_day', columns='days_distance', values='id')


# In[66]:


cohort_counts.head()


# In[67]:


base = cohort_counts[1]


# In[68]:


retention = cohort_counts.divide(base, axis=0).round(3)


# In[74]:


retention


# In[75]:


# Визуализируем результат
# В абсолютных значениях
plt.figure(figsize=(18,14))
plt.title('Users Active')
ax = sns.heatmap(data=cohort_counts, annot=True, vmin=0.0,cmap='Reds')
ax.set_yticklabels(cohort_counts.index)
fig=ax.get_figure()

plt.show()


# In[76]:


# В процентах
plt.figure(figsize=(18,14))
plt.title('Retention Table')
ax = sns.heatmap(data=retention, annot=True, fmt='.0%', vmin=0.0, vmax=1,cmap='Reds')
ax.set_yticklabels(retention.index)
fig=ax.get_figure()

plt.show()


# In[ ]:




