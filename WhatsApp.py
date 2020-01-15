#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[19]:


from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from string import punctuation
import pandas as pd
import emoji


# # Read the exported whatsapp chats

# In[20]:


import os
textfiles = ['./dataset/' + filename for filename in os.listdir('./dataset')]
textfiles


# # Utils-function

# In[21]:


def readfile(textfile):
    file = open(textfile, 'r')
    text = file.readlines()
    file.close()
    return text


# In[22]:


def format(line):
    formatted = dict()
    if ' - ' not in line or ', ' not in line or ': ' not in line:
        return None
    
    try:
        datetime, raw = line.split(' - ', 1)
        date, time = datetime.split(', ', 1)
    
    except ValueError:
        print(line)
        return None
    
    if ': ' in raw:
        linemessage = raw.split(': ', 1)        
        if linemessage[1] == '<Media omitted>\n':
            return None
        
        author = linemessage[0]
        message = linemessage[1]
        return {'Date' : date, 'Time' : time, 'Author' : author, 'Message' : message}
    return None


# In[23]:


def construct_dataframe(data, textfile):
    for line in readfile(textfile):
        formatted = format(line)
        if formatted != None:
            data=data.append(formatted, ignore_index=True)
    return data


# # main

# In[24]:


data = pd.DataFrame(columns = ['Date', 'Time', 'Author', 'Message'])

for file in textfiles:
    data=data.append(construct_dataframe(data, file))

newdata=data.drop(columns=['Date', 'Time'])
newdata.to_csv('./dataset/whatsapp_df.csv', index = False)

