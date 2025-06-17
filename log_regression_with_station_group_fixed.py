import pandas as pd
import statsmodels.api as sm
import numpy as np

# --- データ読み込み ---
df = pd.read_csv("rent_oita_processed.csv")

# --- 必要な変数の抽出と前処理 ---
df = df[["家賃_num", "築年数_num", "面積_num", "徒歩分_num", "駅名"]].dropna()
df = df[df["家賃_num"] > 0]

# --- 駅カテゴリを定義 ---
station_to_group = {
    "大分駅": "中心市街地",
    "西大分駅": "中心市街地",
    "南大分駅": "住宅地",
    "滝尾駅": "住宅地",
    "古国府駅": "住宅地",
    "賀来駅": "住宅地",
    "高城駅": "郊外",
    "鶴崎駅": "郊外",
    "坂ノ市駅": "郊外",
    "中判田駅": "郊外",
    "敷戸駅": "郊外",
    "豊後国分駅": "郊外",
}

# --- 駅カテゴリをマッピング、未知駅は「その他」に分類 ---
df["駅カテゴリ"] = df["駅名"].map(station_to_group).fillna("その他")

# --- 駅カテゴリのダミー変数化（基準カテゴリを除外） ---
df_dummies = pd.get_dummies(df["駅カテゴリ"], prefix="カテゴリ", drop_first=True)

# --- 説明変数の準備 ---
X = pd.concat([df[["築年数_num", "面積_num", "徒歩分_num"]], df_dummies], axis=1)
X = sm.add_constant(X).astype(float)

# --- 目的変数（家賃）の対数変換 ---
y = np.log(df["家賃_num"].astype(float))

# --- モデル構築 ---
model = sm.OLS(y, X).fit()

# --- 予測値の計算 ---
y_pred_log = model.predict(X)

# --- 結果の保存用データフレーム ---
df_output = df.copy()
df_output["y_true_log"] = y
df_output["y_pred_log"] = y_pred_log
df_output["実家賃"] = np.exp(y)
df_output["予測家賃"] = np.exp(y_pred_log)

# --- 実測・予測データを保存 ---
df_output.to_csv("log_regression_grouped_final_result.csv", index=False, encoding="utf-8-sig")

# --- モデルのサマリーを保存 ---
with open("log_regression_grouped_final_summary.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

print("[INFO] 実測家賃・予測家賃を含むファイルを保存しました。")
