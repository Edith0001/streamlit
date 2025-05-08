import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from model_utils import create_sequence_from_date

# 모델 로드 (compile=False)
model = load_model("model.h5", compile=False)
scaler_y = joblib.load("scaler_y.pkl")

st.title("💱 AI가 알려주는 원/달러 환율 예측")
st.markdown("60일치 데이터를 기반으로 선택한 날짜의 환율을 예측합니다.")

input_date = st.date_input("📅 예측하고 싶은 날짜를 선택하세요", datetime(2025, 3, 31))

if st.button("예측하기"):
    try:
        seq = create_sequence_from_date(pd.to_datetime(input_date))
        pred = model.predict(seq)
        pred_inv = scaler_y.inverse_transform(pred)[0][0]
        st.success(f"예측 환율: {pred_inv:,.2f} 원")
    except ValueError as e:
        st.error(str(e))
