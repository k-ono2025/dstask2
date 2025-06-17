# データサイエンス特論第一第２回課題(分析の適用)
## 概要
“駅近”は地方でも有効か？重回帰による大分市の家賃構造分析
本プロジェクトは、大分市内の賃貸物件データをもとに、家賃を予測する重回帰モデルを構築・評価するものです。  
主な処理は以下の通りです：

- SUUMOサイトからの物件データ収集（スクレイピング）
- データの前処理
- 重回帰分析の実施、正規化重回帰分析の実施
- モデル評価指標（RMSEなど）の算出
- 分析結果の可視化と出力

---

## ファイル構成

task2/
├── suumo_scraper_oita_full.py # SUUMOからのデータ収集スクリプト
├── rent_oita_full.csv # スクレイピングによる生データ
├── normalization.py # データの前処理スクリプト
├── rent_oita_processed.csv # 前処理済みデータ
├── log_regression_with_station_group_fixed.py # 重回帰分析実行（対数変換）
├── log_regression_grouped_final_summary.txt # 実行結果サマリ
├── log_regression_grouped_final_result.csv # 実行結果CSV
├── check_vif.py # VIF算出による多重共線性の確認
├── evaluate_model.py # モデル評価用スクリプト（RMSE等）
├── model_evaluation_metrics.csv # 評価指標出力（RMSEなど）
├── plot_log_space.py # 結果の可視化スクリプト
├── vif_result.csv # VIFの出力結果
├── log_regression_standardized.py # 正規化重回帰分析実行
├── log_regression_grouped_standardized_summary.txt　# 実行結果サマリ
├── log_regression_grouped_standardized_result.csv # 実行結果CSV
└── README.md # このファイル
