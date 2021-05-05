#!/usr/bin/env python
# coding: utf-8

# In[19]:


import re
import joblib
import sklearn_crfsuite


# In[20]:


def get_word(s):
    s=re.split(r'[;,.?\s]\s*', s)   # split with delimiters comma, semicolon and space .. followed by any amount of extra whitespace.
    l=[]
    for word in s:
        if(len(word)>0):
            l.append(word)
    return l


# In[21]:


# example="He is stupid."
# example=get_word(example)
# print(example)


# In[22]:


def word2features(sent, i):
    word = sent[i]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# In[23]:


# X = sent2features(example)
# print(X)


# In[24]:


# Save the model as a pickle in a file
#joblib.dump(crf, 'crf_model1.pkl')
  
# Load the model from the file
# def load_model():

  
# # # Use the loaded model to make predictions
# # knn_from_joblib.predict(X_test)


# In[25]:
def get_tmp(tmp,example,y_pred):
    for i in range(0,len(example)):
        tmp.append([example[i],y_pred[i]])
    return tmp


def prediction(s):
    example=get_word(s)
    X = sent2features(example)
    crf_model = joblib.load('crf_model1.pkl') 
    #crf_model._make_predict_fucntion()
    y_pred = crf_model.predict([X])
    tmp=[]
    tmp=get_tmp(tmp,example,y_pred[0])
    return (tmp)
# #print(X)
# print(example)
# print(y_pred[0])


# In[28]:


#prediction("He is a good boy.")


# In[ ]:




