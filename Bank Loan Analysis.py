#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


# In[2]:


app=pd.read_csv(r"E:\data science\DS NOTES\September22\10Sep22\Default Risk Analysis For Banking Domain\application_data.csv")


# In[3]:


app.head()


# In[4]:


previous_app.head()


# In[5]:


# Feature Selection


# In[6]:


app.columns


# In[7]:


app.shape


# In[8]:


msng_info = pd.DataFrame(app.isnull().sum().sort_values(ascending=False)).reset_index() # missing data information 
msng_info.rename(columns={'index':'col_name',0:'null_count'},inplace=True)
msng_info.head()


# In[9]:


msng_info['msng_pct'] = round((msng_info['null_count']/app.shape[0])*100,2) # missing data percentage

msng_info.head()


# In[10]:


msng_col = msng_info[msng_info['msng_pct']>=40]['col_name'].to_list()
# len(msng_col)

app_msng_rmvd = app.drop(labels=msng_col,axis=1)
app_msng_rmvd.shape


# In[11]:


app_msng_rmvd.head()


# In[12]:


flag_col = []

for col in app_msng_rmvd.columns:
    if col.startswith("FLAG_"):
        flag_col.append(col)
        
flag_col
len(flag_col)


# In[13]:


app_msng_rmvd[flag_col].head()


# In[14]:


flag_tgt_col = (app_msng_rmvd[flag_col+['TARGET']])
flag_tgt_col.head()


# In[15]:


sns.countplot(data=flag_tgt_col,x='FLAG_DOCUMENT_19',hue='TARGET')


# In[16]:



plt.figure(figsize=(20,25))
for i,col in enumerate(flag_col):
    plt.subplot(7,4,i+1)
    sns.countplot(data=flag_tgt_col,x=col,hue='TARGET')


# In[17]:


flg_corr = ['FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE',
            'FLAG_EMAIL','TARGET']
flg_corr


# In[18]:


flg_corr_df = app_msng_rmvd[flg_corr]
flg_corr_df.head()


# In[19]:


flg_corr_df.groupby(['FLAG_OWN_CAR']).size()


# In[20]:


flg_corr_df['FLAG_OWN_CAR'] = flg_corr_df['FLAG_OWN_CAR'].replace({'N':0,'Y':1})
flg_corr_df['FLAG_OWN_REALTY'] = flg_corr_df['FLAG_OWN_REALTY'].replace({'N':0,'Y':1})


# In[21]:


flg_corr_df.groupby(['FLAG_OWN_CAR']).size()


# In[22]:


corr_df = round(flg_corr_df.corr(),2)
corr_df


# In[23]:


plt.figure(figsize=(10,5))
sns.heatmap(corr_df,cmap='coolwarm',linewidths=0.5,annot=True)


# In[24]:


app_flg_rmvd = app_msng_rmvd.drop(labels=flag_col,axis=1)

app_flg_rmvd.shape


# In[25]:


app_flg_rmvd.head()


# In[26]:


sns.heatmap(round(app_flg_rmvd[['EXT_SOURCE_2','EXT_SOURCE_3','TARGET']].corr(),2),cmap='coolwarm',linewidths=0.5,annot=True)


# In[27]:


app_source_col_rmvd = app_flg_rmvd.drop(['EXT_SOURCE_2','EXT_SOURCE_3'],axis=1)
app_source_col_rmvd.shape


# # Feature Engineering 
# 
# Missing imputation 
# 
# Value Modification
# 
# Outlier detection and treatment
# 
# Binning
# 
# 

# In[28]:


app_source_col_rmvd.isnull().sum().sort_values()/app_source_col_rmvd.shape[0]


# In[29]:


# Missing  value imputation


# In[30]:


app_source_col_rmvd.groupby(['CNT_FAM_MEMBERS']).size()


# In[31]:


app_source_col_rmvd.CNT_FAM_MEMBERS.mode()


# In[32]:


app_source_col_rmvd['CNT_FAM_MEMBERS'] = app_source_col_rmvd['CNT_FAM_MEMBERS'].fillna((app_source_col_rmvd['CNT_FAM_MEMBERS'].mode()[0]))


# In[33]:


app_source_col_rmvd['CNT_FAM_MEMBERS'].isnull().sum()


# In[34]:


app_source_col_rmvd.groupby(['OCCUPATION_TYPE']).size().sort_values()


# In[35]:


app_source_col_rmvd['OCCUPATION_TYPE'].mode()[0]


# In[36]:


app_source_col_rmvd['OCCUPATION_TYPE'] = app_source_col_rmvd['OCCUPATION_TYPE'].fillna((app_source_col_rmvd['OCCUPATION_TYPE'].mode()[0]))


# In[37]:


app_source_col_rmvd['OCCUPATION_TYPE'].isnull().sum()


# In[38]:


app_source_col_rmvd.info()


# In[39]:


app_source_col_rmvd.groupby(['NAME_TYPE_SUITE']).size()


# In[40]:


app_source_col_rmvd['NAME_TYPE_SUITE'].mode()[0]


# In[41]:


app_source_col_rmvd['NAME_TYPE_SUITE'] = app_source_col_rmvd['NAME_TYPE_SUITE'].fillna(app_source_col_rmvd['NAME_TYPE_SUITE'].mode()[0])


# In[42]:


app_source_col_rmvd['NAME_TYPE_SUITE'].isnull().sum()


# In[43]:


app_source_col_rmvd['AMT_ANNUITY'].describe()


# In[44]:


app_source_col_rmvd['AMT_ANNUITY'] = app_source_col_rmvd['AMT_ANNUITY'].fillna((app_source_col_rmvd['AMT_ANNUITY'].mean()))


# In[45]:


app_source_col_rmvd['AMT_ANNUITY'].isnull().sum()


# In[46]:


app_source_col_rmvd['AMT_REQ_CREDIT_BUREAU_HOUR'].dtype


# In[47]:


app_source_col_rmvd['AMT_REQ_CREDIT_BUREAU_HOUR'].describe()


# In[48]:


app_source_col_rmvd['AMT_REQ_CREDIT_BUREAU_HOUR'].unique()


# In[49]:


app_req_col = []

for col in app_source_col_rmvd.columns:
    if col.startswith('AMT_REQ_CREDIT_BUREAU'):
        app_req_col.append(col)

app_req_col


# In[50]:


for col in app_req_col:
    app_source_col_rmvd[col] = app_source_col_rmvd[col].fillna(app_source_col_rmvd[col].median())


# In[51]:


app_source_col_rmvd[app_req_col].isnull().sum()


# In[52]:


app_source_col_rmvd.isnull().sum().sort_values()


# In[53]:


app_source_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# In[54]:


app_source_col_rmvd['AMT_GOODS_PRICE'].describe()


# In[55]:


app_source_col_rmvd['AMT_GOODS_PRICE'].agg(['min','max','median'])


# In[56]:


app_source_col_rmvd['AMT_GOODS_PRICE'].mean()


# In[57]:


app_source_col_rmvd['AMT_GOODS_PRICE'] = app_source_col_rmvd['AMT_GOODS_PRICE'].fillna(app_source_col_rmvd['AMT_GOODS_PRICE'].median())


# In[58]:


app_source_col_rmvd['AMT_GOODS_PRICE'].isnull().sum()


# In[59]:


app_source_col_rmvd.head()


# # Value Modification

# In[60]:


days_col = []

for col in app_source_col_rmvd.columns:
    if col.startswith('DAYS_'):
        days_col.append(col)
        
days_col


# In[61]:


for col in days_col:
    app_source_col_rmvd[col] = abs(app_source_col_rmvd[col])

# app_source_col_rmvd['DAYS_BIRTH'] = abs(app_source_col_rmvd['DAYS_BIRTH'])


# In[62]:


app_source_col_rmvd.head()


# # Outlier Detection and Treatment

# In[63]:


app_source_col_rmvd.nunique().sort_values()


# In[64]:


app_source_col_rmvd['OBS_30_CNT_SOCIAL_CIRCLE'].unique()


# In[65]:


app_source_col_rmvd['AMT_GOODS_PRICE'].agg(['min','max','median'])


# In[66]:


sns.kdeplot(data=app_source_col_rmvd,x='AMT_GOODS_PRICE')


# In[67]:


sns.boxplot(data=app_source_col_rmvd,x='AMT_GOODS_PRICE')


# In[68]:


app_source_col_rmvd['AMT_GOODS_PRICE'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[69]:


bins =[0,100000,200000,300000,400000,500000,600000,700000,800000,900000,4050000]

ranges = ['0-100K','100K-200K','200K-300K','300K-400K','400K-500K','500K-600K','600K-700K'
         ,'700K-800K','800K-900K','Above 900K']

app_source_col_rmvd['AMT_GOODS_PRICE_RANGE'] = pd.cut(app_source_col_rmvd['AMT_GOODS_PRICE'],bins,labels=ranges)


# In[70]:


app_source_col_rmvd.groupby(['AMT_GOODS_PRICE_RANGE']).size()


# In[71]:


app_source_col_rmvd['AMT_INCOME_TOTAL'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[72]:


bins =[0,100000,150000,200000,250000,300000,350000,400000,472500]

ranges = ['0-100K','100K-150K','150K-200K','200K-250K','250K-300K','300K-350K','350K-400K'
         ,'Above 400K']

app_source_col_rmvd['AMT_INCOME_TOTAL_RANGE'] = pd.cut(app_source_col_rmvd['AMT_INCOME_TOTAL'],bins,labels=ranges)


# In[73]:


app_source_col_rmvd.groupby(['AMT_INCOME_TOTAL_RANGE']).size()


# In[74]:


app_source_col_rmvd['AMT_CREDIT'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[75]:


bins =[0,200000,400000,600000,800000,900000,1000000,1854000]

ranges = ['0-200K','200K-400K','400K-600K','600K-800K','800K-900K','900K-1M'
         ,'Above 1M']

app_source_col_rmvd['AMT_CREDIT_RANGE'] = pd.cut(app_source_col_rmvd['AMT_CREDIT'],bins,labels=ranges)


# In[76]:


app_source_col_rmvd.groupby(['AMT_CREDIT_RANGE']).size()


# In[77]:


app_source_col_rmvd['AMT_ANNUITY'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99])


# In[78]:


app_source_col_rmvd['AMT_ANNUITY'].max()


# In[79]:


bins =[0,25000,50000,100000,150000,200000,258025]

ranges = ['0-25K','25K-50K','50K-100K','100K-150K','150K-200K',
         'Above 200K']

app_source_col_rmvd['AMT_ANNUITY_RANGE'] = pd.cut(app_source_col_rmvd['AMT_ANNUITY'],bins,labels=ranges)


# In[80]:


app_source_col_rmvd.groupby(['AMT_ANNUITY_RANGE']).size()


# In[81]:


app_source_col_rmvd['DAYS_EMPLOYED'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.85,0.9,0.95,0.99])


# In[82]:


app_source_col_rmvd['DAYS_EMPLOYED'].agg(['min','max','median'])


# In[83]:


app_source_col_rmvd['DAYS_EMPLOYED'].max()


# In[84]:


bins =[0,1825,3650,5475,7300,9125,10950,12775,14600,16425,18250,365243]

ranges = ['0-5Y','5Y-10Y','10Y-15Y','15Y-20Y','20Y-25Y','25Y-30Y','30Y-35Y','35Y-40Y','40Y-45Y','45Y-50Y','Above 50Y']

app_source_col_rmvd['DAYS_EMPLOYED_RANGE'] = pd.cut(app_source_col_rmvd['DAYS_EMPLOYED'],bins,labels=ranges)


# In[85]:


app_source_col_rmvd.groupby(['DAYS_EMPLOYED_RANGE']).size()


# In[86]:


app_source_col_rmvd['DAYS_BIRTH'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.85,0.9,0.95,0.99])


# In[87]:


app_source_col_rmvd['DAYS_BIRTH'].min()


# In[88]:


bins =[0,7300,10950,14600,18250,21900,25229]

ranges = ['20Y','20Y-30Y','30Y-40Y','40Y-50Y','50Y-60Y','Above 60Y']

app_source_col_rmvd['DAYS_BIRTH_RANGE'] = pd.cut(app_source_col_rmvd['DAYS_BIRTH'],bins,labels=ranges)


# In[89]:


app_source_col_rmvd.groupby(['DAYS_BIRTH_RANGE']).size()


# # Data Analysis

# In[90]:


app_source_col_rmvd.info()


# In[91]:


app_source_col_rmvd.dtypes.value_counts()


# In[92]:


obj_var = app_source_col_rmvd.select_dtypes(include=['object']).columns
obj_var


# In[93]:


app_source_col_rmvd.groupby(['NAME_CONTRACT_TYPE']).size()


# In[94]:


sns.countplot(data=app_source_col_rmvd,x='NAME_CONTRACT_TYPE',hue='TARGET')


# In[95]:


data_pct = app_source_col_rmvd[['NAME_CONTRACT_TYPE','TARGET']].groupby(['NAME_CONTRACT_TYPE'], as_index=False).mean()
data_pct


# In[96]:


data_pct['PCT'] = data_pct['TARGET']*100
data_pct


# In[97]:


sns.barplot(data=data_pct,x='NAME_CONTRACT_TYPE',y='PCT')


# In[98]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
sns.countplot(data=app_source_col_rmvd,x='NAME_CONTRACT_TYPE',hue='TARGET')
plt.subplot(1,2,2)

sns.barplot(data=data_pct,x='NAME_CONTRACT_TYPE',y='PCT')


# In[99]:


plt.figure(figsize=(25,60))

for i,var in enumerate(obj_var):
    
    data_pct = app_source_col_rmvd[[var,'TARGET']].groupby([var],as_index=False).mean().sort_values(by='TARGET',ascending=False)
    data_pct['PCT'] = data_pct['TARGET']*100
    
    
    plt.subplot(10,2,i+i+1)
    plt.subplots_adjust(wspace=0.1,hspace=1)
    sns.countplot(data=app_source_col_rmvd,x=var,hue='TARGET')
    plt.xticks(rotation=90)
    
    plt.subplot(10,2,i+i+2)
    sns.barplot(data=data_pct,x=var,y='PCT',palette='coolwarm')
    plt.xticks(rotation=90)


# In[100]:


num_var = app_source_col_rmvd.select_dtypes(include=['float64','int64']).columns
num_cat_var = app_source_col_rmvd.select_dtypes(include=['float64','int64','category']).columns
num_var


# In[101]:


num_data = app_source_col_rmvd[num_var]
defaulters = app_source_col_rmvd[app_source_col_rmvd['TARGET']==1].drop(['TARGET'],axis=1)
repayers = app_source_col_rmvd[app_source_col_rmvd['TARGET']==0].drop(['TARGET'],axis=1)
repayers.head()


# In[102]:


defaulters.head()


# In[103]:


defaulters[['SK_ID_CURR','CNT_CHILDREN','AMT_INCOME_TOTAL']].corr()


# In[104]:


defaulters_corr = defaulters.corr()
defaulters_corr_unstack= defaulters_corr.where(np.triu(np.ones(defaulters_corr.shape),k=1).astype('bool')).unstack().reset_index().rename(columns={'level_0':'var1','level_1':'var2',0:'corr'})
defaulters_corr_unstack['corr'] = abs(defaulters_corr_unstack['corr'])
defaulters_corr_unstack.dropna(subset=['corr']).sort_values(by=['corr'],ascending=False).head(10)


# In[105]:


repayers_corr = repayers.corr()
repayers_corr_unstack= repayers_corr.where(np.triu(np.ones(repayers_corr.shape),k=1).astype('bool')).unstack().reset_index().rename(columns={'level_0':'var1','level_1':'var2',0:'corr'})
repayers_corr_unstack['corr'] = abs(repayers_corr_unstack['corr'])
repayers_corr_unstack.dropna(subset=['corr']).sort_values(by=['corr'],ascending=False).head(10)


# In[106]:


amt_value=['AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE']


# In[107]:


amt_value


# In[108]:


sns.kdeplot(data=num_data, x='AMT_INCOME_TOTAL', hue='TARGET')


# In[109]:


plt.figure(figsize=(10,12))

for i,col in enumerate(amt_value):
    plt.subplot(2,2,i+1)
    sns.kdeplot(data=num_data,x=col,hue='TARGET')
    plt.subplots_adjust(wspace=0.5,hspace=0.5)


# In[110]:


num_cat_var


# # Bivariant Analysis 

# In[111]:




sns.scatterplot(data=num_data, x='AMT_GOODS_PRICE', y='AMT_CREDIT', hue='TARGET')


# In[112]:


fig, axs = plt.subplots(len(amt_value)-1, len(amt_value)-1, figsize=(10, 10))
for i in range(len(amt_value)-1):
    for j in range(i+1, len(amt_value)):
        # Scatter plot of two variables
        sns.scatterplot(data=num_data, x=num_data[amt_value[i]], y=num_data[amt_value[j]], ax=axs[i, j-1], hue='TARGET')
        # Add labels and title
        
plt.tight_layout()
plt.subplots_adjust(wspace=0.5, hspace=1)
# Show plot
plt.show()


# In[113]:


num_data[['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE']].corr()


# In[114]:


amt_data = num_data[['AMT_INCOME_TOTAL',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'AMT_GOODS_PRICE','TARGET']]

sns.pairplot(data = amt_data , hue = 'TARGET')

