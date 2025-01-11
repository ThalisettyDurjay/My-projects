#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('ipl_2022_dataset.csv')


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.sample()


# In[6]:


df.drop('Unnamed: 0', axis =1, inplace = True)


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df[df['Cost IN $ (000)'].isnull()]


# In[10]:


df['COST IN ₹ (CR.)'] = df['COST IN ₹ (CR.)'].fillna(0)
df['Cost IN $ (000)'] = df['Cost IN $ (000)'].fillna(0)


# In[11]:


df[df['2021 Squad'].isnull()]


# In[12]:


df['2021 Squad'] = df['2021 Squad'].fillna('Not Participated in IPL 2021')


# In[13]:


df.isnull().sum()


# In[14]:


teams = df[df['COST IN ₹ (CR.)']>0]['Team'].unique()
teams


# In[15]:


df['status'] = df['Team'].replace(teams,'sold')


# In[16]:


df['Base Price'].unique()


# In[17]:


df['retention'] = df['Base Price']


# In[18]:


df['retention'].replace(['2 Cr', '40 Lakh', '20 Lakh', '1 Cr', '75 Lakh',
       '50 Lakh', '30 Lakh','1.5 Cr'],'In Auction', inplace = True)


# In[19]:


df['Base Price'].replace('Draft Pick',0, inplace = True)


# In[20]:


df['base_price_unit'] = df['Base Price'].apply(lambda x: str(x).split(' ')[-1])
df['base_price'] = df['Base Price'].apply(lambda x: str(x).split(' ')[0])


# In[21]:


df['base_price'].replace('Retained',0,inplace=True)
df['base_price_unit'].unique()


# In[22]:


df['base_price_unit'] = df['base_price_unit'].replace({'Cr':100,'Lakh':1,'Retained':0})
df['base_price'] = df['base_price'].astype(float)
df['base_price_unit'] = df['base_price_unit'].astype(int)


# In[23]:


df['base_price'] = df['base_price']*df['base_price_unit']


# In[24]:


df.head()


# In[25]:


df.drop(['Base Price','base_price_unit'], axis =1, inplace = True)
df


# In[26]:


df['COST IN ₹ (CR.)'] = df['COST IN ₹ (CR.)']*100


# In[27]:


df = df.rename(columns={'TYPE':'Type','COST IN ₹ (CR.)':'Sold_for_lakh','Cost IN $ (000)':'Cost_in_dollars','2021 Squad':'Prev_team','Team':'Curr_team'})


# In[28]:


df.head()


# In[29]:


df[df['Player'].duplicated(keep=False)]


# In[30]:


df.shape[0]


# In[31]:


types = df['Type'].value_counts()
types.reset_index()


# In[32]:


plt.pie(types.values, labels=types.index,labeldistance=1.2,autopct='%1.2f%%')
plt.title('Role of Players Participated', fontsize = 15)
plt.plot()


# In[33]:


plt.figure(figsize=(5,5))
fig = sns.countplot(df['status'],palette=['Green','Red'])
plt.xlabel('Sold or Unsold')
plt.ylabel('Number of Players')
plt.title('Sold vs Unsold', fontsize=15)
plt.plot()

for p in fig.patches:
    fig.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width()/2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')


# In[34]:


df.sample()


# In[35]:


plt.figure(figsize=(20,10))
fig = sns.countplot(df[df['Curr_team']!='Unsold']['Curr_team'])
plt.xlabel('Name of Team')
plt.ylabel('Number of Players')
plt.title('Players Brought by each Team', fontsize=15)
plt.xticks(rotation=90)
plt.plot()

for p in fig.patches:
    fig.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width()/2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')


# In[36]:


df.groupby(['Curr_team','retention'])['retention'].count()[:-1]


# In[37]:


df.groupby(['Type','status'])['Player'].count().reset_index()


# In[38]:


df.replace({'SRH':'Sunrisers Hyderabad','CSK':'Chennai Super Kings','MI':'Mumbai Indians','KKR':'Kolkata Knight Riders','RR':'Rajasthan Royals','PBKS':'Punjab Kings','DC':'Delhi Capitals','RCB':'Royal Challengers Bangalore'},inplace =True)


# In[39]:


same_team = df[(df['Curr_team']==df['Prev_team']) & (df['retention']=='In Auction')]
same_team


# In[40]:


same_team[same_team.Curr_team=='Royal Challengers Bangalore']


# In[41]:


plt.figure(figsize=(10,8))
sns.countplot(same_team['Curr_team'])
plt.title('Players Who brough by their 2021 teams in Auction ')
plt.xlabel('Name of Team')
plt.ylabel('Number of Player')
plt.xticks(rotation = 90)
plt.grid(axis='y')
plt.plot()


# In[42]:


plt.figure(figsize=(20,10))
fig = sns.countplot(df[df['Curr_team']!='Unsold']['Curr_team'],hue=df['Type'])
plt.title('Players in Each Team')
plt.xlabel('Name of Team')
plt.ylabel('Number of Player')


plt.xticks(rotation = 60)

for p in fig.patches:
    fig.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width()/2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 4), textcoords = 'offset points')


# In[43]:


df[df['retention']=='In Auction'].groupby(['Curr_team'])['Sold_for_lakh'].max()[:-1].sort_values(ascending = False)


# In[44]:


df[(df['retention']=='In Auction') & (df['Type']=='BATTER')].sort_values(by='Sold_for_lakh', ascending = False).head(5)


# In[45]:


df[df['retention']=='Retained'].sort_values(by = 'Sold_for_lakh', ascending = False).head(1)


# In[46]:


amount_spent = df.groupby('Curr_team')['Sold_for_lakh'].sum()[:-1]
amount_spent


# In[47]:


plt.figure(figsize=(15,5))
sns.barplot('Curr_team','Sold_for_lakh', data = df[df['Curr_team']!='Unsold'])
plt.xticks(rotation=60)
plt.ylabel('Ammount Spent')
plt.legend()


# In[48]:


unsold_stars = df[(df.Prev_team != 'Not Participated in IPL 2021') & (df.Curr_team == 'Unsold')][['Player','Prev_team']]


# In[49]:


unsold_stars


# In[ ]:




