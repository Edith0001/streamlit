import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
import joblib
from tensorflow.keras.models import load_model
from model_utils import create_sequence_from_date

# ✅ Streamlit 설정
st.set_page_config(page_title="내일의 환율", page_icon="💸")

# ✅ 모델 및 스케일러 로드
model = load_model("model.h5", compile=False)
scaler_y = joblib.load("scaler_y.pkl")

st.title("💸 내일의 환율")
st.markdown("60일치 데이터를 기반으로 환율을 예측합니다.")

# ✅ 날짜 입력
min_date = date(2025, 1, 1)
max_date = date(2025, 4, 30)
input_date = st.date_input(
    "📅 예측하고 싶은 날짜를 선택하세요",
    value=date(2025, 2, 28),
    min_value=min_date,
    max_value=max_date
)

# ✅ 예측 버튼 실행
if st.button("예측하기"):
    with st.spinner("모델이 환율을 계산 중입니다... 💻"):
        try:
            seq = create_sequence_from_date(pd.to_datetime(input_date))
            pred = model.predict(seq)
            pred_inv = scaler_y.inverse_transform(pred)[0][0]
            st.success(f"📈 예측 환율: **{pred_inv:,.2f} 원**")
        except ValueError as e:
            st.error(str(e))

# ✅ 최근 14일 예측 그래프 (2025-02-15 ~ 2025-02-28)
st.markdown("### 📊 최근 14일간 환율 예측값")

try:
    preds = []
    base_date = date(2025, 2, 28)
    dates = pd.date_range(end=base_date, periods=14)

    for d in dates:
        try:
            seq = create_sequence_from_date(d)
            pred = model.predict(seq)
            pred_inv = scaler_y.inverse_transform(pred)[0][0]
            preds.append((d, pred_inv))
        except:
            continue

    if preds:
        pred_df = pd.DataFrame(preds, columns=["date", "예측 환율"])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(pred_df["date"], pred_df["예측 환율"], marker='o')

        ax.set_xlabel("date", fontsize=11)
        ax.set_ylabel("KRW/USD", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("예측 가능한 날짜가 부족합니다.")
except:
    st.error("📉 그래프 생성 중 오류가 발생했습니다.")
