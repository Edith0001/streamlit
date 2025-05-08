import pandas as pd
import numpy as np
import joblib

def create_sequence_from_date(input_date, seq_len=60):
    df = pd.read_csv("all_features.csv", index_col=0, parse_dates=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    df['환율_lag1'] = df['원/미국달러(매매기준율)'].shift(1)
    df['환율_ma3'] = df['원/미국달러(매매기준율)'].rolling(3).mean()
    df['환율_pct'] = df['원/미국달러(매매기준율)'].pct_change()
    df['환율_diff'] = df['원/미국달러(매매기준율)'].diff()

    for col in ['금', '은', '원유', 'KOSPI', 'NASDAQ']:
        df[f'{col}_pct'] = df[col].pct_change()
        df[f'{col}_diff'] = df[col].diff()

    for col in ['국고채(3년)(%)', '국고채(10년)(%)', '기준금리', 'CD(91일)', '무담보콜금리(1일, 전체거래)']:
        df[f'{col}_diff'] = df[col].diff()

    for col in ['수출물가지수(달러기준)', '수입물가지수(달러기준)']:
        df[f'{col}_pct'] = df[col].pct_change()

    for col in ['뉴스심리지수', '경제심리지수', '미국 소비자심리지수']:
        df[f'{col}_diff'] = df[col].diff()

    df = df.dropna()
    selected_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['lag', 'ma', 'pct', 'diff'])]

    if input_date not in df.index:
        raise ValueError("해당 날짜는 데이터에 없습니다.")
    idx = df.index.get_loc(input_date)
    if idx < seq_len:
        raise ValueError("예측하려는 날짜에 필요한 과거 60일치 데이터가 부족합니다.")

    X_last = df[selected_cols].iloc[idx-seq_len:idx]

    scaler_X = joblib.load("scaler_X.pkl")
    X_scaled = scaler_X.transform(X_last)
    return X_scaled.reshape(1, seq_len, -1)
