import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import tree
import streamlit as st
from web_function import train_model

from sklearn.model_selection import train_test_split

def app(df, x, y):
  warnings.filterwarnings("ignore")
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.title("Visualisasi Data")
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
  if st.checkbox("Plot Confusion Matrix"):
    model, score = train_model(x,y)
    plt.figure(figsize=(10,10))
    pred = model.predict(x_test)
    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    st.pyplot()
  
  if st.checkbox("Plot Decision Tree"):
    model, score = train_model(x,y)
    dot_data = tree.export_graphviz(
      decision_tree=model,  max_depth=5, out_file=None, filled=True, rounded=True, feature_names=x.columns, 
      class_names=["tidak bembeli", "beli"], 
    )
    st.graphviz_chart(dot_data)
