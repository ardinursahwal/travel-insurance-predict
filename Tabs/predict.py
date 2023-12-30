import streamlit as st
from web_function import predict

def app(df, x, y):
    st.title("Prediksi")
    st.write("Sebuah Perusahaan Tour & Travels Menawarkan Paket Asuransi Perjalanan Kepada Pelanggannya.")

    yes_no_data = {0: "Tidak", 1: "Ya"}
    employment_data = {0:"Private Sector/Self Employed", 1:"Government Sector"}

    col1, col2 = st.columns(2)
    with col1: 
        Age = st.number_input("Umur", 25,35,29)
    with col2:
        Employment = st.selectbox("Tipe Pekerjaan", options=list(employment_data.keys()), format_func=lambda x:employment_data[x]) 
    with col1 :    
        Graduate = st.selectbox("Apakah lulusan perguruan tinggi", options=list(yes_no_data.keys()), format_func=lambda x:yes_no_data[x]) 
    with col2:
        Income = st.number_input("Pendapatan tahunan dalam Rupee India",300000,1800000,1500000) 
    with col1:
        Family = st.number_input("Jumlah anggota dalam keluarga pelanggan.", 2,9,3) 
    with col2 :  
        Chronic = st.selectbox("Apakah memiliki kondisi kronis", options=list(yes_no_data.keys()), format_func=lambda x:yes_no_data[x])

    with col1:
        Frequent = st.selectbox("Apakah sering memesan tiket pesawat", options=list(yes_no_data.keys()), format_func=lambda x:yes_no_data[x]) 
    with col2:
        EverTravelled = st.selectbox("Apakah penah berpergian keluar negeri", options=list(yes_no_data.keys()), format_func=lambda x:yes_no_data[x])
    features = [Age,Employment,Graduate,Income,Family,Chronic,Frequent,EverTravelled]

    if st.button("Prediksi untuk membeli asuransi"):
      pred, score = predict(x, y, features)
      st.info("Prediksi Sukses...")
      if(pred[0]==0):
          st.write("Pelanggan tidak akan membeli asuransi")
      else : st.success("Pelanggan akan membeli asuransi")
      st.write("Akurasi Model : ", (score*100), "%")