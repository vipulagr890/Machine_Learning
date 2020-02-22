#!/usr/bin/env python
# coding: utf-8

# In[191]:


import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[213]:


data = pd.read_csv('heart_disease_dataset.csv')


# In[214]:


data.describe()


# In[215]:


data.max()


# In[216]:


chol = DataFrame(data, columns=['chol'])
AGE = DataFrame(data, columns=['age'])


# In[217]:


regr = LinearRegression()
regr.fit(AGE, chol)


# In[227]:


plt.figure(figsize=(15,6))
plt.scatter(AGE, chol, alpha='0.5', s = 200)
plt.scatter(AGE, regr.predict(AGE), linewidth = '2',color = 'k', s = 100, alpha = 0.3)
plt.title('AGE OF PERSON VS CHOLESTROL', fontsize = '20', color='r',)
plt.ylabel('Cholestrol Value', fontsize='20')
plt.xlabel('Age of Patient', fontsize='20')
plt.ylim(100, 600)
plt.xlim(25, 80)
plt.show()


# In[230]:


regr.score(AGE, chol) * 100


# In[220]:


regr.coef_[0][0]


# In[221]:


regr.intercept_[0]


# In[222]:


CHOL= float(181.533) + float((1.12)*(35))
CHOL


# In[223]:


data


# In[224]:


print('Theta1, which is the slope:    ', regr.coef_[0][0])
print('the intercept is:    ', regr.intercept_[0])
print('the goodness of fit:   ', regr.score(AGE, chol))
Predictedchol = regr.predict(AGE)
print(Predictedchol)
print(chol)


# In[ ]:





# In[ ]:




