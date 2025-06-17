import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- フォント（日本語が崩れる場合は適宜変更） ---
plt.rcParams['font.family'] = 'Hiragino Sans'  # Mac
# plt.rcParams['font.family'] = 'MS Gothic'   # Windowsの場合

# --- データ読み込み ---
df = pd.read_csv("log_regression_grouped_final_result.csv")

# --- 残差（対数）計算 ---
df["log_residual"] = df["y_true_log"] - df["y_pred_log"]

# --- プロット①：log(実家賃) vs log(予測家賃) ---
plt.figure(figsize=(6, 6))
plt.scatter(df["y_true_log"], df["y_pred_log"], alpha=0.6)
plt.plot([df["y_true_log"].min(), df["y_true_log"].max()],
         [df["y_true_log"].min(), df["y_true_log"].max()],
         'r--', label='y = x')
plt.xlabel("log(実家賃)")
plt.ylabel("log(予測家賃)")
plt.title("対数スケール：実測 vs 予測")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("log_true_vs_log_predicted.png")
plt.close()

# --- プロット②：log(予測家賃) vs log残差 ---
plt.figure(figsize=(8, 5))
plt.scatter(df["y_pred_log"], df["log_residual"], alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("log(予測家賃)")
plt.ylabel("log(残差)")
plt.title("対数スケール：予測値 vs 残差（log）")
plt.grid(True)
plt.tight_layout()
plt.savefig("log_residual_vs_log_predicted.png")
plt.close()

# --- プロット③：log残差ヒストグラム ---
plt.figure(figsize=(8, 5))
plt.hist(df["log_residual"], bins=30, edgecolor='black')
plt.title("log残差のヒストグラム")
plt.xlabel("log(残差)")
plt.ylabel("件数")
plt.grid(True)
plt.tight_layout()
plt.savefig("log_residual_histogram.png")
plt.close()

print("[INFO] 対数スケールでの可視化が完了しました。")
