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

# ✅ 단일 날짜 예측 버튼
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

# ✅ 🔮 3~5월(92일) rolling 예측 추가
st.markdown("### 🔮 2025년 3~5월 환율 전망 (Rolling Prediction)")

try:
    future_days = 92  # 3/1 ~ 5/31
    last_60_X = X_scaled[-60:].copy()
    predicted_rates = []

    for _ in range(future_days):
        pred = model.predict(last_60_X.reshape(1, 60, -1), verbose=0)
        pred_rate = scaler_y.inverse_transform(pred)[0][0]
        predicted_rates.append(pred_rate)

        # 입력 업데이트 (예측값 기반이므로 동일 피처 구조 유지)
        next_input = last_60_X[-1].copy()
        last_60_X = np.vstack([last_60_X[1:], next_input])

    # 변화율 계산
    predicted_returns = [0]
    for i in range(1, len(predicted_rates)):
        change = (predicted_rates[i] - predicted_rates[i-1]) / predicted_rates[i-1] * 100
        predicted_returns.append(change)

    # 데이터프레임 구성
    future_dates = pd.date_range(start="2025-03-01", periods=future_days)
    rolling_df = pd.DataFrame({
        "DATE": future_dates,
        "예측 환율": predicted_rates,
        "예측 변화율 (%)": predicted_returns
    })

    # 그래프 시각화
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(rolling_df["DATE"], rolling_df["예측 환율"], label="예측 환율", color="blue", marker='o')
    ax2.set_xlabel("DATE")
    ax2.set_ylabel("KRW/USD")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax2.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("2025년 3~5월 환율 예측 (LSTM rolling)")
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("📄 예측값 보기"):
        st.dataframe(rolling_df.set_index("DATE"), use_container_width=True)

except Exception as e:
    st.error(f"🚨 Rolling 예측 오류: {e}")
