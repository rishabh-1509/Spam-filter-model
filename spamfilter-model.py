#Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Importing the dataset
#replacing thr null values with null string
df = pd.read_csv('mail_data.csv')
x= df.iloc[: , : -1].values
y = df.iloc[: , -1].values
print(df)
print(y)
mail_data =  df.where((pd.notnull(df)),'')
#checking number of rows and coloum in dataset
mail_data.shape
#label encodeing spam = 1, ham =0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x = le.fit_transform(x)
#splitting the Dataset into train test 
X_train , X_test, y_train, y_test = train_test_split(y,x, test_size = 0.2, random_state= 1)
#transform the text data into feature vectors
feature_extraction =  TfidfVectorizer(min_df =1 , stop_words = 'english', lowercase = True)
X_train_features =  feature_extraction.fit_transform(X_train )
X_test_feature = feature_extraction.transform(X_test )
# convert y_train
y_train = y_train.astype('int')
y_test = y_test.astype('int')
#training based on logistic regression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_features,y_train)
#predicting the value Y
y_pred = classifier.predict(X_train_features)
acc_on_train = accuracy_score(y_train,y_pred)
print(acc_on_train)
#testing the model on a sample mail from the DATASET 
input_mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
input_data_featurs = feature_extraction.transform(input_mail)
y_pres = classifier.predict(input_data_featurs)
print(y_pres)