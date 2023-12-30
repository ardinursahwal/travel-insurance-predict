import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

@st.cache_data()
def load_data():
    df = pd.read_csv('TravelInsurancePrediction.csv')
    # hapus kolom yang tidak diperlukan
    df.drop(["Unnamed: 0"], axis=1, inplace=True)

    # ubah objek menjadi numerikal
    df['GraduateOrNot'] = df['GraduateOrNot'].map({'Yes': 1, 'No': 0})
    df['FrequentFlyer'] = df['FrequentFlyer'].map({'Yes': 1, 'No': 0})
    df['EverTravelledAbroad'] = df['EverTravelledAbroad'].map({'Yes': 1, 'No': 0})
    # Government Sector : 1, Private Sector/Self Employed : 0
    df["Employment Type"] = df["Employment Type"].map({"Government Sector" : 1, "Private Sector/Self Employed" : 0})
    
    x = df.drop(['TravelInsurance'],axis=True)
    y = df['TravelInsurance']
    return df, x, y

@st.cache_data()
def train_model(x,y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
  mscale=MinMaxScaler()
  mscale.fit_transform(x_train)
  mscale.transform(x_test)
  # Melakukan pemangkasan (post-pruning) menggunakan ccp_alpha=0.013
  model = DecisionTreeClassifier(random_state=0, ccp_alpha=0.013)
  model.fit(x_train, y_train)

  y_pred = model.predict(x_test)
  score = accuracy_score(y_test, y_pred)

  return model, score

def predict(x, y, features):
  model, score = train_model(x,y)
  pred = model.predict(np.array(features).reshape(1,-1))
  return pred, score
