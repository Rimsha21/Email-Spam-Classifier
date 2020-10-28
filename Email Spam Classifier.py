
# coding: utf-8

# # EMAIL SPAM CLASSIFIER MODEL BY Rimsha Virmani S15

# In[44]:


import pandas as pd
import numpy as np


# In[45]:


df=pd.read_csv("spam.csv",encoding='ISO-8859-1')


# In[46]:


df.head()


# In[47]:


df['spam']=df['v1'].apply(lambda x:1 if x=='spam' else 0)


# In[48]:


df.head()


# In[68]:


columns=['Unnamed:  2','Unnamed:  3','Unnamed:  4']
df.drop(columns,axis=1,inplace=True)


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


labelencoder= LabelEncoder()


# In[18]:


df['spam']=labelencoder.fit_transform(df['v1'])


# In[19]:


df.head()


# In[20]:


from sklearn.model_selection import train_test_split


# In[54]:


x_train,x_test,y_train,y_test= train_test_split(df.v2,df.spam,test_size=0.2)


# In[22]:


x_train.shape


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer


# In[56]:


cv = CountVectorizer()


# In[58]:


x_train_count= cv.fit_transform(x_train.values)
x_train_count.toarray()[:3]


# In[62]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_count,y_train)


# In[63]:


x_test_count=cv.transform(x_test)


# In[78]:


model.score(x_test_count,y_test)


# In[79]:


emails=['Hello Faisal , we are looking for machine learning intern','Discount upto 30% on First Purchase']
email_count=cv.transform(emails)


# In[80]:


model.predict(email_count)


# In[81]:


email=['Win an iphone 11']
email_count=cv.transform(email)
model.predict(email_count)

