import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import tree
import streamlit as st
from web_function import train_model

def app(df, x, y):
  warnings.filterwarnings("ignore")
  st.set_option('deprecation.showPyplotGlobalUse', False)

  st.title("Visualisasi Data")

  if st.checkbox("Plot Confusion Matrix"):
    model, score = train_model(x,y)
    plt.figure(figsize=(10,10))
    pred = model.predict(x)
    cm = confusion_matrix(y, pred, labels=model.classes_)
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