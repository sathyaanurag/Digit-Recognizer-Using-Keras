
# coding: utf-8

# In[1]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:


from keras.utils import np_utils
from sklearn.model_selection import KFold


# In[3]:


seed = 7
numpy.random.seed(seed)


# In[4]:


train = pandas.read_csv("train.csv")


# In[5]:


# load dataset
dataset = train.values
dataset.shape


# In[6]:


#trying with small fraction of data
trainfraction = train.sample(frac=0.1)


# In[7]:


dataset = trainfraction.values


# In[8]:


Y = dataset[:,0:1].astype(float)


# In[9]:


# split into input (X) and output (Y) variables
X = dataset[:,1:784].astype(float)


# In[10]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[12]:


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(783, input_dim=783, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[13]:


#estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
estimator = KerasClassifier(build_fn=baseline_model, verbose=0)


# In[14]:


kfold = KFold(n_splits=2, shuffle=True, random_state=seed)


# In[15]:


results = cross_val_score(estimator, X, dummy_y, cv=kfold)


# In[16]:


print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:




