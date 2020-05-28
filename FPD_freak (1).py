#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imorting necessary libraries
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 


# In[ ]:


#using pandas to read the training and test data
data = pd.read_excel("/content/train.xlsx")


# In[6]:


#displaying the first and last 5 rows of the data
data


# In[8]:


data.count()


# 
# ###### the training data has 10,683 rows but 'Route', 'Total_Stops' have only 10,682 rows
# ###### This shows that there are some *missing values* in the dataset

# ##**Starting the preliminary analysation of data**

# In[9]:


data.dtypes


# In[10]:


data.describe(include = "all")


# In[11]:


data.info


# ##**Identifying and dealing with Missing values**

# In[12]:


data.isnull()


# In[13]:


#To count how many missing values are there in the dataset in row as well as column
data.isnull().sum().sum()


# In[14]:


#To count how many missing values are there in the dataset column wise
data.isnull().sum()


# # **Data Cleaning**

# In[24]:


data = data.fillna(training_data['Route'].value_counts().index[0])
data.isnull().sum().sum()


# ##### It shows that we have cleaned our data

# ## These factors directly affect airline ticket prices:
# 
# ##### **1. Airline**
# ##### **2. Date of Journey**
# ##### **3. Route**
# ##### **4. Duration**
# ##### **5. Total stops**

# #**Data Formatting**
# 

# In[ ]:


# Dropping all unnecessary columns
data = data.drop(['Source','Destination','Dep_Time','Arrival_Time','Additional_Info'], axis=1)


# In[26]:


data.dtypes


# #### Now we can see that Airline, Date_of_Journey, Route, Duration and Total Stops are not numerical or integer data type

# ##Formatting Airlines column

# In[27]:


# All different unique airlines
data['Airline'].unique()


# ##### Here you can find who had the most number of bookings.

# In[28]:


# Showing details according to airlines

filter = data['Airline']=="Jet Airways"
data.where(filter).count()


# In[ ]:


# Encoding airline categorical values to numerical values
le = LabelEncoder()
data['Airline'] = le.fit_transform(data['Airline'])


# In[30]:


data['Airline'].unique()


# In[31]:


data.head()


# In[32]:


# To verify Jet Airways is now termed as 4

filter = data['Airline']==4
data.where(filter).count()


# ### Formatting Date_of_Journey column
# ###### We need to split it into date, month, and year

# In[ ]:


data['Date'] = data['Date_of_Journey'].str.split('/').str[0]
data['Month'] = data['Date_of_Journey'].str.split('/').str[1]
data['Year'] = data['Date_of_Journey'].str.split('/').str[2]


# In[ ]:


# Now convert them into int type
data['Date'] = data['Date'].astype(int)
data['Month'] = data['Month'].astype(int)
data['Year'] = data['Year'].astype(int)


# In[ ]:


# Drop original date_of_journey column
data = data.drop(['Date_of_Journey'], axis=1)


# In[36]:


data.head()


# ### Formatting Duration
# ###### We need to deal with alphabet 'h' and 'm' alongwith whitespace

# In[37]:


D = []
for i in data['Duration']:
    for j in range(0,len(i)):
        i=str(i).replace('h','')
        i=str(i).replace('m','')
        i = str(i).replace(' ','')
    D.append(i)
data['Duration_Mins'] = D
data.head()


# In[38]:


data.dtypes
#he type of Duration_Mins is object


# In[ ]:


# Converting object type to integer data type
data['Duration_Mins'] = data['Duration_Mins'].astype(int)


# In[40]:


# Finally converting hours into minutes

for i in range(0,len(data['Duration_Mins'])):
  if len(str(data['Duration_Mins'][i]))<=2:
    data['Duration_Mins'][i]*=60
  else:
    new=0
    s=''
    res = list(map(int, str(data['Duration_Mins'][i])))
    new=res[0]*60
    res.pop(0)
    for j in res:
      s=s+str(j)
    m=int(s)
    new = new + m
    data['Duration_Mins'][i]=new


# In[ ]:


# Dropping original Duration column
data = data.drop(['Duration'], axis=1)


# In[42]:


data.head()


# ###Formatting Total_Stops column

# In[43]:


for i in range(len(data['Total_Stops'])):
  if data['Total_Stops'][i]=='non-stop':
    data['Total_Stops'][i]='0'
  elif data['Total_Stops'][i]=='2 stops':
    data['Total_Stops'][i]='2'
  else:
    data['Total_Stops'][i]='1'


# In[ ]:


# Converting object type to integer data type
data['Total_Stops'] = data['Total_Stops'].astype(int)


# In[45]:


data.head()


# ### Formatting Route column
# ##### Here I have used length function to distinguish if route is direct then its value is 0

# In[49]:


# To know the different values so that I convert them easily to 3 different categories

for i in range(len(training_data['Route'])):
  print(len(str(training_data['Route'][i])))
  if i==10:
    break


# In[ ]:


for i in range(len(data['Route'])):
  if len(str(data['Route'][i]))==9:
    data['Route'][i]=0
  elif len(str(data['Route'][i]))==15:
    data['Route'][i]=1
  else:
    data['Route'][i]=2


# In[ ]:


# Converting object type to integer data type
data['Route'] = data['Route'].astype(int)


# In[52]:


data.dtypes


# ## Data Visualization

# In[72]:


data.head()


# In[62]:


data['Airline'].value_counts().plot(kind='barh', color='brown')


# ##### This shows that Jet Airways (labelled as 4) has the max number of bookings

# In[59]:


ax = data.plot.scatter('Airline',y='Price', figsize=(20,6))
plt.xlabel('Airlines')
plt.ylabel('Price of Airlines')
plt.show()


# In[64]:


ax1 = data.plot.scatter('Duration_Mins','Price', color='yellow')


# ##### From the above graph it is quite clear that flights with less travel time i.e. Duration_Mins are booked by maximum number of users.

# In[69]:


ax1 = data.plot.scatter('Route','Price')


# In[73]:


ax1 = data.plot.scatter('Total_Stops','Price', color='pink')


# ##### It is very obvious that flight with 0 stop or non-stop flights are cheaper than with 1 or 2 stops.

# In[61]:


ax1 = data.plot.scatter('Date','Price',color='green')


# # Data Normalization

# In[ ]:


training_data.head()


# ### Train-Test-Split
# 

# In[ ]:


y = data['Price']
x = data.drop(['Price'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=4)


# ### Faeture Scaling

# In[76]:


mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(x_train)
mm_scaler.transform(x_test)


# ### PCA

# In[ ]:


# Make an instance of the Model
pca = PCA()


# In[ ]:


x_train = pd.DataFrame(pca.fit_transform(x_train))
x_test = pd.DataFrame(pca.transform(x_test))


# In[ ]:


explained_variance = pca.explained_variance_ratio_ 


# In[80]:


explained_variance


# # Data modelling and prediction

# #### Random forest

# In[84]:


# create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(x_train, y_train)   


# In[ ]:


y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)


# In[97]:


display(regressor.score(x_test,y_test))


# In[98]:


display(regressor.score(x_train,y_train))


# In[106]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print('Train data\n')
print("Absolute Error = ",mean_absolute_error(y_pred_train,y_train))
print("Mean percentage error = ",mean_absolute_percentage_error(y_pred_train,y_train))
print("\nTest data\n")
print("Absolute Error = ",mean_absolute_error(yY_pred,y_test))
print("Mean percentage error = ",mean_absolute_percentage_error(yY_pred,y_test))


# #### Linear Regression

# In[92]:


lr = LinearRegression() 
lr.fit(x_train, y_train) 


# In[ ]:


#Predicting the test set result using  
# predict function under LinearRegression
y_pred_test = lr.predict(x_test) 
y_pred_train = lr.predict(x_train)


# In[94]:


len(y_pred)


# In[95]:


display(lr.score(x_train,y_train))


# In[96]:


display(lr.score(x_test,y_test))


# In[108]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print('Train data\n')
print("Absolute Error = ",mean_absolute_error(y_pred_train,y_train))
print("Mean percentage error = ",mean_absolute_percentage_error(y_pred_train,y_train))
print("\nTest data\n")
print("Absolute Error = ",mean_absolute_error(yY_pred,y_test))
print("Mean percentage error = ",mean_absolute_percentage_error(yY_pred,y_test))


# In[ ]:


# Data Normalization
#y = training_data(['Airline', 'Route', 'Total_Stops', ])

#training_data['Price']=((training_data['Price']-training_data['Price'].min())/(training_data['Price'].max()-training_data['Price'].min()))*2
#training_data['Duration_Mins']=((training_data['Duration_Mins']-training_data['Duration_Mins'].min())/(training_data['Duration_Mins'].max()-training_data['Duration_Mins'].min()))*2

#training_data['Route']=((training_data['Route']-training_data['Route'].min())/(training_data['Route'].max()-training_data['Route'].min()))*2
#training_data['Duration_Mins'] = round(training_data['Duration_Mins'],2)

#training_data['Price'] = round(training_data['Price'],2)

#print(training_data['Date'].unique())
#print(training_data['Month'].unique())

#Normalizing the data
#Applying Z-Score inprice column
#training_data['Normalized_Price'] = (training_data['Price']-training_data['Price'].mean())/training_data['Price'].std()
#training_data['Normalized_Price']

# ML Decision Trees

#from sklearn.tree import DecisionTreeClassifier 
#dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 
#dtc.fit(x_train, y_train) 

#y_pred_test = dtc.predict(x_test) 
#y_pred_train = dtc.predict(x_train) 

#from sklearn.metrics import accuracy_score

#print("Confusion Matrix: ", confusion_matrix(y_train, y_pred_train)) 
      
#print ("Accuracy : ", accuracy_score(y_train,y_pred_train)*100)

#from sklearn.linear_model import Ridge
#ridge_model = Ridge(alpha = 0.9)
#ridge_model.fit(x_train,y_train)

#ridge_train_pred = ridge_model.predict(x_train)
#print(mean_absolute_percentage_error(ridge_train_pred,y_train))

#ridge_test_pred = ridge_model.predict(x_test)
#print(mean_absolute_percentage_error(ridge_test_pred,y_test))


# In[ ]:




