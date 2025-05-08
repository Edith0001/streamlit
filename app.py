import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ✅ 페이지 설정
st.set_page_config(page_title="내일의 환율", page_icon="💸")
st.title("💸 내일의 환율")
st.markdown("60일치 데이터를 기반으로 환율을 예측합니다.")

# ✅ 모델 및 스케일러 불러오기
model = load_model("model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ✅ 데이터 로드 및 전처리
df = pd.read_csv("all_features.csv", index_col=0, parse_dates=True)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.replace(",", "")
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

target_col = "원/미국달러(매매기준율)"
df["환율_lag1"] = df[target_col].shift(1)
df["환율_ma3"] = df[target_col].rolling(3).mean()
df["환율_pct"] = df[target_col].pct_change()
df["환율_diff"] = df[target_col].diff()

for col in ["금", "은", "원유", "KOSPI", "NASDAQ"]:
    df[f"{col}_pct"] = df[col].pct_change()
    df[f"{col}_diff"] = df[col].diff()

for col in ["국고채(3년)(%)", "국고채(10년)(%)", "기준금리", "CD(91일)", "무담보콜금리(1일, 전체거래)"]:
    df[f"{col}_diff"] = df[col].diff()

for col in ["수출물가지수(달러기준)", "수입물가지수(달러기준)"]:
    df[f"{col}_pct"] = df[col].pct_change()

for col in ["뉴스심리지수", "경제심리지수", "미국 소비자심리지수"]:
    df[f"{col}_diff"] = df[col].diff()

df = df.dropna()
selected_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['lag', 'ma', 'pct', 'diff'])]
X_scaled = scaler_X.transform(df[selected_cols])

# ✅ Rolling prediction: 3~5월 (92일)
last_60_X = X_scaled[-60:].copy()
predicted_rates = []
future_days = 92

for _ in range(future_days):
    pred = model.predict(last_60_X.reshape(1, 60, -1), verbose=0)
    pred_rate = scaler_y.inverse_transform(pred)[0][0]
    predicted_rates.append(pred_rate)
    next_input = last_60_X[-1].copy()
    last_60_X = np.vstack([last_60_X[1:], next_input])

# ✅ 변화율 계산
predicted_returns = [0]
for i in range(1, len(predicted_rates)):
    change = (predicted_rates[i] - predicted_rates[i - 1]) / predicted_rates[i - 1] * 100
    predicted_returns.append(change)

# ✅ 결과 DataFrame
future_dates = pd.date_range(start="2025-03-01", periods=future_days)
rolling_df = pd.DataFrame({
    "DATE": future_dates,
    "예측 환율": predicted_rates,
    "예측 변화율 (%)": predicted_returns
})

# ✅ 날짜 선택 시 예측값 출력
st.markdown("### 📆 날짜 선택하여 예측 환율 확인")
selected_date = st.date_input("날짜를 선택하세요 (2025-03-01 ~ 2025-05-31)", value=date(2025, 3, 1),
                              min_value=date(2025, 3, 1), max_value=date(2025, 5, 31))

if selected_date in rolling_df["DATE"].values:
    row = rolling_df[rolling_df["DATE"] == pd.to_datetime(selected_date)].iloc[0]
    rate = row["예측 환율"]
    change = row["예측 변화율 (%)"]
    st.success(f"📅 {selected_date.strftime('%Y-%m-%d')}의 예측 환율은 **{rate:,.2f}원**입니다.\n\n"
               f"전일 대비 변화율: **{change:+.2f}%**")
else:
    st.warning("선택한 날짜에 대한 예측값이 없습니다.")

# ✅ 그래프 시각화
st.markdown("### 🔮 2025년 3~5월 환율 예측 그래프")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(rolling_df["DATE"], rolling_df["예측 환율"], label="예측 환율", marker='o')
ax.set_xlabel("DATE")
ax.set_ylabel("KRW/USD")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.tick_params(axis='x', labelrotation=45, labelsize=8)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_title("2025년 3~5월 환율 예측 (Rolling Prediction)")
st.pyplot(fig)
plt.close(fig)

with st.expander("📄 예측 수치 보기"):
    st.dataframe(rolling_df.set_index("DATE"), use_container_width=True)
