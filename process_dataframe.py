#!/usr/bin/env python
# coding: utf-8

# In[5]:


# your name mentioned in your whatsapp account at the time of exporting chats
import json
with open('./myname.json') as json_file:
    data = json.load(json_file)
my_name = data['name']


# # Imports

# In[5]:


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


# # Stopwords listing
# ## Extend the list of stopwords by analyzing your language transiliterated in english (ex-hindi, bengali, telegu, tamil etc) for better precision

# In[6]:


stopwords = nltk.corpus.stopwords.words('english')

# Transliterated Hindi Words
stopwords.extend(['hai','nhi','mai','toh','ho','kya','na','ka','hi','ki','tum','nahi','bhi',
                  'haan','se','ke','tha','k','aur','rhe','ko','rhi','main','mujhe','abhi','voh','b',
                  'hun','thi','hain','ek','kar','rha','e','hoga','kal','lekin','tumne',
                  'hua','arey','pr','koi','liye','hum','maine','gaya','accha','aa','tumhe','mera',
                  'kuch','yeh','hota','u','ye','time','bohot','er','tumhara','lab',
                  'kyun','kr','class','fir','sir','hu','gayi','karna','chahiye','acha','n','jo','nt'])

le = WordNetLemmatizer()


# # Function to preprocess the data

# In[7]:


def preprocess(text):   
    
    tokens=word_tokenize(text.lower())
    
    punctuation=re.compile('[^a-zA-Z]*')    
    post_punctutation=([punctuation.sub("", word) for word in tokens])
      
    stem_token=[le.lemmatize(word) for word in post_punctutation if word not in stopwords]    
    return " ".join(stem_token) 


# 
# # Loading csv file

# In[ ]:


df = read_csv('./dataset/whatsapp_df.csv')

# preprocess the messages
df['Message']=df['Message'].transform(lambda x : preprocess(x))

# drop rows where message is too small
message_threshold_size = 4
df=df[df['Message'].apply(lambda x : len(word_tokenize(x)))>=message_threshold_size]


# In[10]:


# drop your own messages as it may overshadow others' messages due to high occurence
# comment it out if you want to include your chats
df=df[df['Author'] != my_name]

df.head()


# In[11]:


df=df.sample(frac=1).reset_index(drop=True)
df.to_csv('./dataset/processed.csv')

