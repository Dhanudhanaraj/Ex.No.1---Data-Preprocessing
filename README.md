# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## REQUIPMENTS REQUIRED:

Hardware – PCs

Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :

Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:

1.Importing the libraries.

2.Importing the dataset.

3.Taking care of missing data.

4.Encoding categorical data.

5.Normalizing the data.

6.Splitting the data into test and train.

## PROGRAM:
```
Developed By:D.Dhanumalya.
Register Number:212222230030.

import pandas as pd
df=pd.read_csv("/content/Churn_Modelling (1).csv")
df.head()

df.duplicated()

df.describe()

df.isnull().sum()

x=df.iloc[:, :-1].values
print(x)

y=df.iloc[:, -1].values
print(y)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)

```
## OUTPUT:
![NN1](https://user-images.githubusercontent.com/119218812/229430046-3454376c-0b1f-407f-a4ea-9139d395e748.png)

![NN2](https://user-images.githubusercontent.com/119218812/229430061-12468285-27fb-4b01-969f-daf92a8fa1d0.png)

![NN3](https://user-images.githubusercontent.com/119218812/229430073-59d3580f-a68d-4bfb-82df-79f38ca67585.png)

![NN4](https://user-images.githubusercontent.com/119218812/229430089-1fa82fc5-363a-4563-b614-748adf0f592f.png)

![NN5](https://user-images.githubusercontent.com/119218812/229430109-2f80cabd-b8c5-4366-9930-77738c3cb840.png)

![NN6](https://user-images.githubusercontent.com/119218812/229430132-ef53a8c2-fea2-4ba3-b33d-0dfb575f4ea7.png)

![NN7](https://user-images.githubusercontent.com/119218812/229430145-77965388-2c79-4dc9-9efc-60de9331c017.png)

![NN8](https://user-images.githubusercontent.com/119218812/229430161-3f702c27-7edd-405c-b224-6ede279d3a79.png)

![NN9](https://user-images.githubusercontent.com/119218812/229430176-681ee055-ad0e-4575-815c-e2cda4f2bc07.png)

## RESULT:
Thus the above program for standardizing the given data was implemented successfully.

