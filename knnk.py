#!/usr/bin/env python
# coding: utf-8

# # In the name of Allah

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from numpy import dot
from numpy.linalg import norm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import keras.backend as K
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, load_model, save_model, Model
from keras.models import Sequential
from keras.layers import Dense
import pickle
import joblib
import re


# In[4]:

with open('pos.pkl', 'rb') as f:
        pos = pickle.load(f)
with open('neg.pkl', 'rb') as f:
    neg = pickle.load(f)

a_file = open("oneDEmbeddings.pkl", "rb")
oneDEmbeddings = pickle.load(a_file)


a_file = open("embeddings.pkl", "rb")
embeddings = pickle.load(a_file)
    
    

def get_word(s):
    s=re.split(r'[;,.?\s]\s*', s)   # split with delimiters comma, semicolon and space .. followed by any amount of extra whitespace.
    l=[]
    for word in s:
        if(len(word)>0):
            l.append(word)
    return l


# In[16]:


def getEmbeddings(embeddings,word):
    a=np.zeros(50)
    if word not in embeddings:
        return a
    return embeddings[word]


# In[17]:


def getEmbeddings2(oneDEmbeddings,word):
    a=0
    if word not in oneDEmbeddings:
        return a
    return oneDEmbeddings[word][0]


# In[18]:


def getKNearest2(oneDwordEmbedding,pos,tmp,k):
    low,high=0,len(pos)-1
    ans=0
    while low<=high:
        mid=int((low+high)/2)
        if pos[mid]==oneDwordEmbedding:
            ans=mid
            break
        elif pos[mid]<oneDwordEmbedding:
            ans=mid
            low=mid+1
        else:
            high=mid-1
        
    cnt,i,j=0,ans-1,ans+1
    while cnt<10 and i>=0 and j<len(pos):
        cnt+=1
        if abs(pos[i]-oneDwordEmbedding)<=abs(pos[j]-oneDwordEmbedding):
            tmp.append(pos[i])
            i-=1
        else:
            tmp.append(pos[j])
            j+=1
                
        while cnt<10 and j<len(pos):
            cnt+=1
            tmp.append(pos[j])
            j+=1
            
        while cnt<10 and i>=0:
            cnt+=1
            tmp.append(pos[i])
            i-=1


# In[19]:


def cossim(wordEmbedding,pos):
    p1=pos
    p1=p1.reshape(-1)
    w1=wordEmbedding
    w1=w1.reshape(-1)
    if norm(p1)==0 or norm(w1)==0:
        return 0
    val=dot(w1,p1)/(norm(w1)*norm(p1))
    return val


# In[116]:


def get_feature(naive_bayes_model,y_pred,example,k1):
    
    for i in range(0,len(example)):
        tmp=[]
        word=example[i].lower()
        wordEmbedding=getEmbeddings(embeddings,word)
        oneDwordEmbedding=getEmbeddings2(oneDEmbeddings,word)
        getKNearest2(oneDwordEmbedding,pos,tmp,k1)
        getKNearest2(oneDwordEmbedding,neg,tmp,k1)
        for k in range(i-1,max(-1,i-6),-1):
            l=y_pred[k]
            newEmbedding=getEmbeddings(embeddings,example[k].lower())
            tmp.append(cossim(wordEmbedding,newEmbedding))
            tmp.append(l)
        while len(tmp)<2*k1+10:
            tmp.append(0)
        
        ans=naive_bayes_model.predict([tmp])
        
        if ans[0][0]<0.5:
            ans[0][0]=0.0
        else:
            ans[0][0]=1.0
        y_pred.append(ans[0][0])


# In[115]:


def get_feature1(naive_bayes_model,y_pred,example,k1):

    for i in range(0,len(example)):
        tmp=[]
        word=example[i].lower()
        wordEmbedding=getEmbeddings(embeddings,word)
        oneDwordEmbedding=getEmbeddings2(oneDEmbeddings,word)
        getKNearest2(oneDwordEmbedding,pos,tmp,k1)
        getKNearest2(oneDwordEmbedding,neg,tmp,k1)
        for k in range(i-1,max(-1,i-6),-1):
            l=y_pred[k]
            newEmbedding=getEmbeddings(embeddings,example[k].lower())
            tmp.append(cossim(wordEmbedding,newEmbedding))
            tmp.append(l)
        while len(tmp)<2*k1+10:
            tmp.append(0)
        
        ans=naive_bayes_model.predict([tmp])
        
        y_pred.append(ans[0])


# In[45]:


def get_tmp(tmp,example,y_pred):
    for i in range(0,len(example)):
        if y_pred[i]<0.5:
            y_pred[i]='A'
        else:
            y_pred[i]='P'
        tmp.append([example[i],y_pred[i]])
    return tmp


# In[120]:


def naive_bayes_prediction(s):
    example=get_word(s)
    naive_bayes_model = joblib.load('naive_bayes_model.pkl')
    y_pred=[]
    get_feature1(naive_bayes_model,y_pred,example,10)
    tmp=[]
    get_tmp(tmp,example,y_pred)
    return (tmp)


# In[123]:


# tmp=naive_bayes_prediction("He is a idiot.")
# print(tmp)


# In[118]:


def random_forest_prediction(s):
    example=get_word(s)
    random_forest_model = joblib.load('clf_model_real.pkl')
    y_pred=[]
    get_feature1(random_forest_model,y_pred,example,10)
    tmp=[]
    get_tmp(tmp,example,y_pred)
    return tmp


# In[122]:


# tmp=random_forest_prediction("He is a idiot.")
# print(tmp)
#print(y_pred)


# In[62]:


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# In[109]:


def ann_prediction(s):
    example=get_word(s)
    dependencies={
        'f1': f1
    }
    ann_model = keras.models.load_model("ann_model.h5" , custom_objects=dependencies)
    y_pred=[]
    get_feature(ann_model,y_pred,example,10)
    tmp=[]
    get_tmp(tmp,example,y_pred)
    return (tmp)


# In[110]:


# tmp=ann_prediction("He is a idiot.")
# print(tmp)


# In[ ]:




