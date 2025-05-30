# -*- coding: utf-8 -*-
"""UTS Model Deployment_Main OOP

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eBs-1_5niSZd9SVfSoaZk6SyR9GPDpDq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

import pickle

class Preprocessor:
  def __init__(self, filepath):
    self.filepath= filepath
    self.df= None

  def read_data(self):
    self.df= pd.read_csv(self.filepath)

  def drop_identifier_column(self):
    self.df= self.df.drop(columns=["Booking_ID"])

  def check_duplicate(self):
    print("\nTotal duplicated data before:", self.df.duplicated().sum())

  def remove_duplicate(self):
    self.df = self.df.drop_duplicates()
    print("\nTotal duplicated data after removing:",self.df.duplicated().sum())

  def checking_null(self):
    print("Total null values:\n", self.df.isnull().sum())

  def remove_null(self):
    self.df['avg_price_per_room'] = self.df['avg_price_per_room'].fillna(self.df['avg_price_per_room'].median())
    self.df['required_car_parking_space'] = self.df['required_car_parking_space'].fillna(self.df['required_car_parking_space'].median())
    self.df['type_of_meal_plan'] = self.df['type_of_meal_plan'].fillna(self.df['type_of_meal_plan'].mode()[0])
    print("Total null values after:\n", self.df.isnull().sum())

  def anomalies(self):
    self.df['booking_status'] = self.df['booking_status'].replace('Canceled', 'Cancelled')
    self.df['booking_status'] = self.df['booking_status'].replace('Not_Canceled', 'Not_Cancelled')

  def encoding(self):
    cate = []
    nume = []

    for i in self.df.columns:
      if self.df[i].dtype == 'object':
          cate.append(i)
      else:
          nume.append(i)

    self.df = pd.get_dummies(self.df, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=False)
    self.df['booking_status'] = self.df['booking_status'].map({'Not_Cancelled': 0, 'Cancelled': 1})

  def define_x_y(self):
    x = self.df.drop('booking_status', axis = 1)
    y = self.df['booking_status']
    return x,y

class Modeling:
  def __init__(self,
               x,
               y,
               n_estimators = 200,
               learning_rate = 0.8,
               max_depth = 3,
               eval_metric='logloss',
               random_state=7):
    self.x= x
    self.y= y
    self.x_train, self.x_test, self.y_train, self.y_test, self.y_pred= [None] * 5
    self.model= XGBClassifier(n_estimators = n_estimators,
                                       learning_rate = learning_rate,
                                       max_depth = max_depth,
                                       eval_metric = eval_metric,
                                       random_state = random_state)

  def split_data(self, test_size= 0.2, random_state= 7):
    self.x_train, self.x_test, self.y_train, self.y_test= train_test_split(self.x, self.y, test_size= test_size, random_state= random_state)

  def scaling(self):
    scaler = RobustScaler()

    self.x_train = scaler.fit_transform(self.x_train)
    self.x_test = scaler.transform(self.x_test)

  def train(self):
    self.model.fit(self.x_train, self.y_train)

  def evaluate(self):
    self.y_pred= self.model.predict(self.x_test)
    print("\nClassification Report\n")
    print(classification_report(self.y_test, self.y_pred))
    cm = confusion_matrix(self.y_test, self.y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Cancelled', 'Cancelled'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

  def model_save(self, filepath):
    with open(filepath, "wb") as f:
      pickle.dump(self.model, f)


# -----------
preprocessor= Preprocessor("Dataset_B_hotel.csv")
preprocessor.read_data()
preprocessor.drop_identifier_column()
preprocessor.check_duplicate()
preprocessor.remove_duplicate()
preprocessor.checking_null()
preprocessor.remove_null()
preprocessor.anomalies()
preprocessor.encoding()
x, y= preprocessor.define_x_y()

modeling= Modeling(x, y)
modeling.split_data()
modeling.scaling()
modeling.train()
modeling.evaluate()
modeling.model_save("xgbclassifier_model.pkl")