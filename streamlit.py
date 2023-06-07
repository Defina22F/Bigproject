import pickle
import numpy as np
import streamlit as st
from PIL import Image

#load save model
rfc=pickle.load(open('RFC_model.pkl','rb'))
sc=pickle.load(open('Scaler_model.pkl','rb'))

#judul web
st.title("Prediksi Obesitas dengan RFC")

#untuk input data
col1, col2=st.columns(2)
with col1:
    Age=st.text_input("Umur")
    if Age != '':
        Age = float(Age)  # Konversi ke float
with col2:
    Height=st.text_input("Tinggi Badan")
    if Height != '':
        Height = float(Height)  # Konversi ke float
with col1:
    Weight=st.text_input("Berat Badan")
    if Weight != '':
         Weight= float(Weight)  # Konversi ke float
with col2:
    FCVC=st.text_input("Jumlah Makan sayur dalam sehari")
    if FCVC != '':
         FCVC= float(FCVC)  # Konversi ke float
with col1:
    NCP=st.text_input("Jumlah makan dalam sehari")
    if NCP != '':
        NCP = float(NCP)  # Konversi ke float
with col2:
    CH2O=st.text_input("Minum dalam sehari (Liter)")
    if CH2O != '':
        CH2O = float(CH2O)  # Konversi ke float
with col1:
    FAF=st.text_input("Olahraga dalam seminggu")
    if FAF != '':
        FAF = float(FAF)  # Konversi ke float
with col2:
    TUE=st.text_input("Lama waktu didepan laptop (Jam)")
    if TUE != '':
        TUE = float(TUE)  # Konversi ke float
with col1:
    Gender=st.radio("Gender",['Pria','Wanita'])
    if Gender == 'Wanita':
        Gender = 0.0
    else:
        Gender = 1.0
with col2:
    family_history_with_overweight=st.radio("Riwayat Obesitas Keluarga",['Ya','Tidak'])
    if family_history_with_overweight == 'Tidak':
        family_history_with_overweight = 0.0
    else:
        family_history_with_overweight = 1.0
with col1:
    FAVC=st.radio("Makan Tinggi Kalori",['Ya','Tidak'])
    if FAVC == 'Tidak':
        FAVC = 0.0
    else:
        FAVC = 1.0
with col2:
    CAEC=st.radio("Jumlah Mengemil dalam 1 Hari",['Terkadang','Sering','Selalu','Tidak'])
    if CAEC == 'Tidak':
        CAEC = 0.0
    elif CAEC == 'Selalu':
        CAEC = 1.0
    elif CAEC == 'Sering':
        CAEC = 2.0
    elif CAEC == 'Terkadang':
        CAEC = 3.0
with col1:
    SCC=st.radio("Apakah Anda memantau kalori pada setiap makan",['Tidak','Ya'])
    if SCC == 'Tidak':
        SCC = 0.0
    else:
        SCC = 1.0
with col2:
    CALC=st.radio("Minum Alkohol",['Terkadang','Sering','Selalu','Tidak'])
    if CALC == 'Tidak':
        CALC = 0.0
    elif CALC == 'Selalu':
        CALC = 1.0
    elif CALC == 'Sering':
        CALC = 2.0
    elif CALC == 'Terkadang':
        CALC = 3.0
with col1:
    MTRANS=st.radio("Transportasi",['Transportasi_publik','Automobile','Jalan','Motor','Sepeda'])
    if MTRANS == 'Sepeda':
        MTRANS = 0.0
    elif MTRANS == 'Motor':
        MTRANS = 1.0
    elif MTRANS == 'Jalan':
        MTRANS = 2.0
    elif MTRANS == 'Automobile':
        MTRANS = 3.0
    elif MTRANS == 'Transportasi_publik':
        MTRANS = 4.0
with col2:
    SMOKE=st.radio("Merokok",['NO','YES'])
    if SMOKE == 'NO':
        SMOKE = 0.0
    else:
        SMOKE = 1.0

#kode untuk predikisi
Prediksi_Nobayes =''
if st.button("PREDIKSI SEKARANG"):
    # Mengubah argumen menjadi array numpy dua dimensi
    s=sc.transform([[Age,Height,Weight,FCVC,NCP,CH2O,FAF,TUE]])
    # Melakukan prediksi dengan XGBoost
    Prediksi = rfc.predict([[s[0][0],s[0][1], s[0][2],s[0][3],s[0][4],s[0][5],s[0][6],s[0][7],Gender,family_history_with_overweight,FAVC,CAEC,SCC,CALC,MTRANS,SMOKE]])
    
    if Prediksi[0]==0:
        Prediksi_Nobayes ="Insufficient_Weight"
    elif Prediksi[0] == 1:
        Prediksi_Nobayes = "Normal_Weight"
    elif Prediksi[0] == 2:
        Prediksi_Nobayes = "Obesity_Type_I"
    elif Prediksi[0] == 3:
        Prediksi_Nobayes = "Obesity_Type_II"
    elif Prediksi[0] == 4:
        Prediksi_Nobayes = "Obesity_Type_III"
    elif Prediksi[0] == 5:
        Prediksi_Nobayes = "Overweight_Level_I"
    elif Prediksi[0] == 6:
        Prediksi_Nobayes = "Overweight_Level_II"
    else:
        Prediksi_Nobayes = "tidak ditemukan jenis"

st.success(Prediksi_Nobayes)

#teks
st.caption('Developer')
st.caption('Defina')
st.caption('Meryam')
st.caption('Andi')
