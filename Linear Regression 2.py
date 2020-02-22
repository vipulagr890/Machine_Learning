#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[7]:


data = pd.read_csv('lsd-math-score-data (1).csv')


# In[8]:


data


# In[9]:


X = DataFrame(data, columns=['Drug_ppm'])
y = DataFrame(data, columns=['Math_score'])


# In[10]:


regression = LinearRegression()
regression.fit(X, y)


# In[11]:


plt.figure(figsize=(15,6))
plt.scatter(X ,y)
plt.plot(X ,regression.predict(X))
plt.title('DrugPPM_VS_MathScore')
plt.xlabel('LSD')
plt.ylabel('score')
plt.show()


# In[12]:


regression.coef_


# In[13]:


regression.coef_[0]


# In[14]:


regression.coef_[0][0]


# In[15]:


regression.intercept_


# In[16]:


X = DataFrame(data, columns=['Time_delay'])
y = DataFrame(data, columns=['Drug_ppm'])


# In[17]:


time = data[['Time_delay']]
LSD = data[['Drug_ppm']]
score = data[['Math_score']]


# In[31]:


plt.figure(figsize=(15,6))
plt.scatter(time, LSD, alpha='0.7', linewidth='4', s=200)
plt.text(x = 200, y = 5, s = 'vipul\'s plot' ,fontsize=20)
plt.title('LSD Concentration in PPM VS Time in MInute')
plt.xlabel('Time in Minute')
plt.ylabel('LSD CONCENTRATION IN PPM')
plt.plot(time, LSD, color='r')
plt.show()


# In[19]:


plt.figure(figsize=[15,6])
plt.plot(time, LSD, color='red', linewidth='4', alpha=0.7)
plt.scatter(time, LSD, color='red',  s = 200)
plt.title('LSD Concentration in PPM VS Time in MInute',  fontsize = 20)
plt.xlabel('Time in Minute', fontsize = 20)
plt.ylabel('LSD CONCENTRATION IN PPM',  fontsize = 20)
plt.text(x = 200, y = 5, s = 'Vipul\'s plot', fontsize = 10)
plt.xlim(0,500)
plt.ylim(1,7)
plt.yticks( fontsize = 20)
plt.xticks( fontsize = 20)
plt.show()


# In[20]:


regression.fit(LSD, score)


# In[21]:


regression.coef_


# In[22]:


regression.coef_[0][0] #theta1


# In[23]:


regression.intercept_


# In[24]:


regression.intercept_[0]  #theta0


# In[34]:


print('Theta1, which is the slope:    ', regression.coef_[0][0])
print('the intercept is:    ', regression.intercept_[0])
print('the goodness of fit:   ', regression.score(LSD, score))
PredictedScore = regression.predict(LSD)
print(PredictedScore)
print(score)


# In[ ]:




