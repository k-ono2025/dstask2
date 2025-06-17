import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# --- データ読み込み ---
df = pd.read_csv("rent_oita_processed.csv")

# --- 必要な変数の抽出と前処理 ---
df = df[["家賃_num", "築年数_num", "面積_num", "徒歩分_num", "駅名"]].dropna()
df = df[df["家賃_num"] > 0]

# --- 駅カテゴリのマッピング ---
station_to_group = {
    "大分駅": "中心市街地",
    "西大分駅": "中心市街地",
    "南大分駅": "住宅地",
    "滝尾駅駅": "住宅地",
    "古国府駅": "住宅地",
    "賀来駅": "住宅地",
    "高城駅": "郊外",
    "鶴崎駅": "郊外",
    "坂ノ市駅": "郊外",
    "中判田駅": "郊外",
    "敷戸駅": "郊外",
    "豊後国分駅": "郊外",
}
df["駅カテゴリ"] = df["駅名"].map(station_to_group).fillna("その他")

# --- ダミー変数化 ---
df_dummies = pd.get_dummies(df["駅カテゴリ"], prefix="カテゴリ", drop_first=True)

# --- 説明変数 ---
X_raw = pd.concat([df[["築年数_num", "面積_num", "徒歩分_num"]], df_dummies], axis=1)

# --- 説明変数の標準化 ---
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_raw)
X_scaled = pd.DataFrame(X_scaled_array, columns=X_raw.columns, index=X_raw.index)

# 定数項の追加
X_scaled = sm.add_constant(X_scaled)

# --- 対数変換した目的変数 ---
y = np.log(df["家賃_num"])

# --- 回帰分析実行 ---
model_std = sm.OLS(y, X_scaled).fit()

# --- 結果表示 ---
print(model_std.summary())

# --- 結果保存 ---
result_df = pd.DataFrame({
    "変数名": model_std.params.index,
    "標準化係数": model_std.params.values,
    "標準誤差": model_std.bse.values,
    "t値": model_std.tvalues.values,
    "p値": model_std.pvalues.values
})
result_df.to_csv("log_regression_grouped_standardized_result.csv", index=False, encoding="utf-8-sig")

with open("log_regression_grouped_standardized_summary.txt", "w", encoding="utf-8") as f:
    f.write(model_std.summary().as_text())

print("[INFO] 標準化済みの回帰モデルを保存しました。")
