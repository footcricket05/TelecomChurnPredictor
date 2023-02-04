#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction in Telecom Industry Using Logistic Regression.
# 
# The Customer Churn table contains information on all 7,043 customers from a Telecommunications company in California in Q2 2022
# 
# 
# Each record represents one customer, and contains details about their demographics, location, tenure, subscription services, status for the quarter (joined, stayed, or churned), and more!
# 
# The Zip Code Population table contains complimentary information on the estimated populations for the California zip codes in the Customer Churn table

# # Importing necessary files

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('telecom_customer_churn.csv')


# In[3]:


df.head(5)


# ## Overviewing all the columns in the dataset

# In[4]:


df.columns


# In[5]:


df1 = df.copy()


# Creating a copy of the Dataset

# In[6]:


df1.head(7)


# In[7]:


df1.columns


# # Exploratory Data Analysis

# ## Data Preprocessing

# ## Dropping unwanted columns from the dataset

# In[8]:


df1.drop(['Customer ID','Total Refunds','Zip Code','Latitude', 'Longitude','Churn Category', 'Churn Reason'],axis='columns',inplace=True)


# In[9]:


df1.shape


# In[10]:


df1.dtypes


# Checking the number of unique values in each column

# In[11]:


features = df1.columns
for feature in features:
     print(f'{feature}--->{df[feature].nunique()}')


# ## Getting the percentge of Null Values in each Column

# In[12]:


df1.isnull().sum() / df1.shape[0]


# Cleaning Function for the Dataset

# In[13]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[14]:


df1=df1.interpolate()


# In[15]:


df1=df1.dropna()
df.head()


# In[16]:


df['Unlimited Data'] 


# In[17]:


number_columns=['Age','Number of Dependents','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Extra Data Charges','Total Long Distance Charges','Total Revenue']


# ## Checking the unique values of column having datatype: 'object'

# In[18]:


def unique_values_names(df):
    for column in df:
        if df[column].dtype=='object':
            print(f'{column}:{df[column].unique()}')


# In[19]:


unique_values_names(df1)


# # Data Visualization

# In[20]:


import plotly.express as px 


# ## Visualizing Column 'Age' in the dataset

# In[21]:


fig = px.histogram(df1, x = 'Age')
fig.show()


# ## Checking the stats in number_columns of the copied dataset

# In[22]:


df1.hist(figsize=(15,15), xrot=30)


# In[23]:


df1['Age']


# In[24]:


import matplotlib.pyplot as plt


# ## Visualizing the number of customers who churned, stayed or joined in the company with a bar plot

# In[25]:


Customer_Stayed=df1[df1['Customer Status']=='Stayed'].Age
Customer_Churned=df1[df1['Customer Status']=='Churned'].Age
Customer_Joined=df1[df1['Customer Status']=='Joined'].Age

plt.xlabel('Age')
plt.ylabel('Customers Numbers')
plt.hist([Customer_Stayed,Customer_Churned,Customer_Joined], color=['black','red','blue'],label=['Stayed','Churned','Joined'])

plt.title('Customers Behavior ',fontweight ="bold")
plt.legend()


# In[26]:


import seaborn as sns


#  ## Defining Correlation between the columns in the dataset

# In[27]:


data  = df1.corr()
plt.figure(figsize = (20,10))
sns.heatmap(data, annot = True)


# ## Analyzing Outlier in the dataset with respect to customer status

# In[28]:


fig, ax = plt.subplots(4,3, figsize = (15,15))
for i, subplot in zip(number_columns, ax.flatten()):
    sns.boxplot(x = 'Customer Status', y = i , data = df1, ax = subplot)


# In[29]:


fig = px.density_heatmap(df1, x='Age', y='Total Charges')
fig.show()


# In[30]:


df1.columns


# In[31]:


pd.crosstab(df['Customer Status'], df['Married']).plot(kind='bar')


# In[32]:


pd.crosstab(df['Customer Status'], df['Gender']).plot(kind='bar')


# In[33]:


df1['Payment Method'].unique()


# ## Create dictionary with role / data key value pairs

# In[34]:


Roles = {}
for j in df1['Payment Method'].unique():
    Roles[j] = df1[df1['Payment Method'] == j]


# In[35]:


Roles.keys()


# ## Selecting the rows where the role is 'Credit Card'

# In[36]:


Roles['Credit Card']


# In[37]:


len(Roles)


# ## Checking the number of Offers in the dataset

# In[38]:


off = df1['Offer'].value_counts()
off


# In[39]:


import plotly.graph_objects as go


# In[40]:


fig = go.Figure([go.Bar(x=off.index, y=off.values)])
fig.show()


# In[41]:


df1_off = Roles['Credit Card'].Offer.value_counts()
df1_off


# In[42]:


fig = go.Figure([go.Bar(x= df1_off.index, y=df1_off.values)])
fig.show()


# In[43]:


df1 = df1.rename(columns = {'Customer Status':'Customer_Status'})


# In[44]:


Roles1 = {}
for k in df1['Customer_Status'].unique():
    Roles1[k] = df1[df1['Customer_Status'] == k]
Roles1.keys()


# In[45]:


df1_state = Roles1['Stayed'].Offer.value_counts()
df1_state


# # Data Modelling

# ## Replacing the Gender column in the dataset with Label Encoding
# 
# 0 for Female
# 
# 1 for Male

# In[46]:


df1.replace({"Gender":{'Female':0,'Male':1}},inplace=True)


# ## Replacing the columns with 'yes' and 'no' output by Label Encoding
# 
# 0 for No
# 
# 1 for Yes

# In[47]:


yes_and_no=[  'Paperless Billing', 'Unlimited Data', 
       'Streaming Movies', 'Streaming Music',  'Streaming TV',
       'Premium Tech Support', 'Device Protection Plan', 'Online Backup', 'Online Security',
       'Multiple Lines',  'Married']
for i in yes_and_no:
    df1.replace({'No':0,'Yes':1},inplace=True)


# ## Replacing 'Phone Service' with '1'

# In[48]:


df1.replace({"Phone Service":{'Yes':1}},inplace=True)


# In[49]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1.Customer_Status = le.fit_transform(df1.Customer_Status)


# In[50]:


df1 = pd.get_dummies(data=df1, columns=['Payment Method','Contract','Internet Type','Offer','City'])


# In[51]:


cols_to_scale = ['Age','Number of Dependents','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge', 'Total Charges',
       'Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])


# ## Dealing with Imbalance Data

# ## Dropping the Customer_Status
# 
# i.e. The column tht we have to predict and set as a dependent variable

# In[52]:


X = df1.drop('Customer_Status',axis='columns')
y = df1['Customer_Status']


# In[53]:


X.head(5)


# In[54]:


y.head(5)


# # Data Model Building

# ## Spliting the data in Training and Test Data

# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=5)


# In[56]:


len(X_train)


# In[57]:


X_train[:10]


# ## Importing the required files for the model that is to applied
# 
# 1. Random Forest Classifier
# 2. Logistic Regression
# 3. GaussianNB
# 4. Decision Tree Classifier
# 5. XGB Classifier
# 

# ## Importing Models

# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[59]:


model_params = {
     
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
        }    
    },
       'XGB_Classifier':{
        'model':XGBClassifier(),
        'params':{
            'base_score':[0.5]
            
        }
    },   
}


# In[60]:


from sklearn.model_selection import ShuffleSplit


# ## Getting the best_score from the applied models

# In[61]:


from sklearn.model_selection import GridSearchCV
scores = []
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False)
    clf.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# ## It was concluded that XGB_Classifier was giving us the best_score in the dataset

# ## Selecting the model with best score for the dataset

# In[62]:


reg=XGBClassifier()
reg.fit(X_train, y_train)


# In[63]:


reg.score(X_test, y_test)


# We got an accuracy of 80.86 percent in the testing dataset

# ## Predicting values from the model build to check the accuracy

# In[64]:


y_predicted = reg.predict(X_test)
y_predicted[:5]


# ## Verifying the actual values with the predicted values

# In[65]:


y_test[:5]


# ## Importing Confusion Matrx 

# In[66]:


import seaborn as sn
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ## Importing Classification Report

# In[70]:


from sklearn.metrics import classification_report


# In[71]:


print(classification_report(y_test, y_predicted))


# In[72]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predicted)


# ## In the end we conclude that the Telecom Customer Churn Prediction was best worked with XGB_Classifier with an accuracy score of 80.86%

# In[ ]:




