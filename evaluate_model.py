import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

plt.rcParams['font.family'] = 'Hiragino Sans'
# --- データ読み込み（ここが重要） ---
df = pd.read_csv("log_regression_grouped_final_result.csv")

# --- 残差の計算 ---
df["残差"] = df["実家賃"] - df["予測家賃"]

# --- モデル評価指標の計算 ---
y_true = df["実家賃"]
y_pred = df["予測家賃"]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# --- 評価結果を保存 ---
metrics_df = pd.DataFrame({
    "評価指標": ["RMSE", "MAPE", "R²"],
    "値": [rmse, mape, r2]
})
metrics_df.to_csv("model_evaluation_metrics.csv", index=False, encoding="utf-8-sig")
print("[INFO] モデル評価指標を保存しました。")

# --- 可視化①：残差ヒストグラム ---
plt.figure()
plt.hist(df["残差"], bins=20, edgecolor="black")
plt.title("残差のヒストグラム")
plt.xlabel("残差（実家賃 - 予測家賃）")
plt.ylabel("件数")
plt.savefig("residual_histogram.png")

# --- 可視化②：Q-Qプロット ---
plt.figure()
stats.probplot(df["残差"], dist="norm", plot=plt)
plt.title("Q-Qプロット")
plt.savefig("residual_qqplot.png")

# --- 可視化③：予測家賃 vs 残差 ---
plt.figure()
plt.scatter(df["予測家賃"], df["残差"], alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("予測家賃")
plt.ylabel("残差")
plt.title("予測家賃 vs 残差")
plt.savefig("residual_vs_predicted.png")

print("[INFO] 残差可視化グラフを保存しました。")
