#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("khwaishsaxena/vehicle-price-prediction-dataset")

print("Path to dataset files:", path)


# In[2]:


import pandas as pd 


# In[5]:


df = pd.read_csv(r"C:\Users\adars\Downloads\study_scores_missing.csv")
print(df.head())


# In[7]:


print(df.isnull().sum())


# In[8]:


# Fill missing 'Hours' with its mean
df['Hours'] = df['Hours'].fillna(df['Hours'].mean())

# Fill missing 'Scores' with its mean
df['Scores'] = df['Scores'].fillna(df['Scores'].mean())


# In[9]:


print(df.isnull().sum())



# In[10]:


df


# In[18]:


import seaborn as sns


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[13]:


import numpy as np


# In[23]:


sns.scatterplot(data=df, x='Hours', y='Scores')
plt.title('Hours vs Scores')
plt.show()


# In[24]:


X = df[['Hours']]  # Feature
y = df['Scores']   # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[26]:


y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# In[27]:


plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')  # regression line
plt.title('Regression Line')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[29]:


#We performed a simple linear regression to predict student scores based on the number of hours studied using a dataset that contained some missing values. After filling missing data with the column averages, we trained a linear model.

"""" The regression model shows a strong positive relationship between study hours and scores.

 Mean Squared Error (MSE): 17.25
(On average, predictions deviate from actual scores by ~4.15 points.)

 R² Score: 0.98
(98% of the variation in scores is explained by the number of hours studied.)

The regression line closely fits the data points, indicating the model’s predictions are highly accurate. This analysis confirms that increased study time is strongly associated with higher scores."""


# In[ ]:




