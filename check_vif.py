import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# データ読み込み
df = pd.read_csv("rent_oita_processed.csv")

# 必要な列だけ抽出＋NaN除去
df = df[["家賃_num", "築年数_num", "面積_num", "徒歩分_num", "駅名"]].dropna()
df = df[df["家賃_num"] > 0]

# 駅カテゴリのマッピング
station_to_group = {
    "大分駅": "中心市街地", "西大分駅": "中心市街地",
    "南大分駅": "住宅地", "滝尾駅": "住宅地", "古国府駅": "住宅地", "賀来駅": "住宅地",
    "高城駅": "郊外", "鶴崎駅": "郊外", "坂ノ市駅": "郊外", "中判田駅": "郊外", "敷戸駅": "郊外", "豊後国分駅": "郊外"
}
df["駅カテゴリ"] = df["駅名"].map(station_to_group).fillna("その他")

# ダミー変数化（カテゴリ変数を数値に変換）
df_dummies = pd.get_dummies(df["駅カテゴリ"], prefix="カテゴリ", drop_first=True)

# 説明変数（定数項も追加）
X = pd.concat([df[["築年数_num", "面積_num", "徒歩分_num"]], df_dummies], axis=1)
X = add_constant(X).astype(float)

# VIFの計算
vif_df = pd.DataFrame()
vif_df["変数名"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 結果出力
vif_df.to_csv("vif_result.csv", index=False, encoding="utf-8-sig")
print(vif_df)
