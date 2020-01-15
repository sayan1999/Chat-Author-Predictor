#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[2]:


from sklearn.linear_model import LogisticRegression
from nltk.corpus import movie_reviews
from pandas import DataFrame, read_csv
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.metrics import classification_report
import nltk
import re


# In[3]:


df=read_csv('./dataset/processed.csv')

# if your dataset is large uncomment the following lines, 
# and tune the capacity_rows variable according to your ram capacity

# import numpy as np
# capacity_rows = 70000
# if len(df) > capacity_rows:
#     remove_n = len(df) - capacity_rows
# drop_indices = np.random.choice(df.index, remove_n, replace=False)
# df = df.drop(drop_indices)


# # Vectorize the words and label the classes

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vect=TfidfVectorizer()
X=vect.fit_transform(df['Message'])
dtm_df=DataFrame(X.toarray(), columns=vect.get_feature_names()) 

label=LabelEncoder()
dtm_df['Author']=label.fit_transform(df['Author'])


# In[5]:


# Release ram
del(df)


# In[6]:


test_size_fraction=0.2
train_size = int(len(dtm_df)*(1-test_size_fraction))


# # Train the logistic regression model

# In[7]:


clf_lr=LogisticRegression(C=150)
clf_lr.fit(dtm_df.loc[0:train_size][vect.get_feature_names()],dtm_df.loc[0:train_size]['Author'])
pred=clf_lr.predict(dtm_df.loc[train_size:][vect.get_feature_names()])
print("The accuracy of Logistic Regression :" , accuracy_score(pred,dtm_df.loc[train_size:]['Author'])) 
print("The classification report is : \n"+classification_report(pred,dtm_df.loc[train_size:]['Author']))


# # Dump python object

# In[9]:


import pickle
with open('./objects/label.obj', 'wb') as label_file:
    pickle.dump(label, label_file)
with open('./objects/model.obj', 'wb') as model_file:
    pickle.dump(clf_lr, model_file)
with open('./objects/vect.obj', 'wb') as vect_file:
    pickle.dump(vect, vect_file)

