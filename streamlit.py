import pickle
import numpy as np
import streamlit as st
from PIL import Image

#load save model
rfc=pickle.load(open('RFC_model.pkl','rb'))
sc=pickle.load(open('Scaler_model.pkl','rb'))

#judul web
st.title("prediksi NObayes dengan RFC")

#untuk input data
col1, col2=st.columns(2)
with col1:
    Age=st.text_input("Age")
    if Age != '':
        Age = float(Age)  # Konversi ke float
with col2:
    Height=st.text_input("Height")
    if Height != '':
        Height = float(Height)  # Konversi ke float
with col1:
    Weight=st.text_input("Weight")
    if Weight != '':
         Weight= float(Weight)  # Konversi ke float
with col2:
    FCVC=st.text_input("FCVC")
    if FCVC != '':
         FCVC= float(FCVC)  # Konversi ke float
with col1:
    NCP=st.text_input("NCP")
    if NCP != '':
        NCP = float(NCP)  # Konversi ke float
with col2:
    CH2O=st.text_input("CH2O")
    if CH2O != '':
        CH2O = float(CH2O)  # Konversi ke float
with col1:
    FAF=st.text_input("FAF")
    if FAF != '':
        FAF = float(FAF)  # Konversi ke float
with col2:
    TUE=st.text_input("TUE")
    if TUE != '':
        TUE = float(TUE)  # Konversi ke float
with col1:
    Gender=st.text_input("Gender")
    if Gender != '':
        Gender = float(Gender)  # Konversi ke float
with col2:
    family_history_with_overweight=st.text_input("family_history_with_overweight")
    if family_history_with_overweight != '':
        family_history_with_overweight = float(family_history_with_overweight)  # Konversi ke float
with col1:
    FAVC=st.text_input("FAVC")
    if FAVC != '':
        FAVC = float(FAVC)  # Konversi ke float
with col2:
    CAEC=st.text_input("CAEC")
    if CAEC != '':
        CAEC = float(CAEC)  # Konversi ke float
with col1:
    SCC=st.text_input("SCC")
    if SCC != '':
        SCC = float(SCC)  # Konversi ke float
with col2:
    CALC=st.text_input("CALC")
    if CALC != '':
        CALC = float(CALC)
with col1:
    MTRANS=st.text_input("MTRANS")
    if MTRANS != '':
        MTRANS = float(MTRANS)
with col2:
    SMOKE=st.text_input("SMOKE")
    if SMOKE != '':
        SMOKE = float(SMOKE)

#kode untuk predikisi
Prediksi_Nobayes =''
if st.button("Prediksi SEKARANG"):
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