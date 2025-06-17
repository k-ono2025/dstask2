# データサイエンス特論第一第２回課題(分析の適用)
## 概要
“駅近”は地方でも有効か？重回帰による大分市の家賃構造分析<br>
本プロジェクトは、大分市内の賃貸物件データをもとに、家賃を予測する重回帰モデルを構築・評価するものです。  <br>
主な処理は以下の通りです：<br>

- SUUMOサイトからの物件データ収集（スクレイピング）<br>
- データの前処理<br>
- 重回帰分析の実施、正規化重回帰分析の実施<br>
- モデル評価指標（RMSEなど）の算出<br>
- 分析結果の可視化と出力<br>

---

## ファイル構成

task2/<br>
├── suumo_scraper_oita_full.py # SUUMOからのデータ収集スクリプト<br>
├── rent_oita_full.csv # スクレイピングによる生データ<br>
├── normalization.py # データの前処理スクリプト<br>
├── rent_oita_processed.csv # 前処理済みデータ<br>
├── log_regression_with_station_group_fixed.py # 重回帰分析実行（対数変換）<br>
├── log_regression_grouped_final_summary.txt # 実行結果サマリ<br>
├── log_regression_grouped_final_result.csv # 実行結果CSV<br>
├── check_vif.py # VIF算出による多重共線性の確認<br>
├── evaluate_model.py # モデル評価用スクリプト（RMSE等）<br>
├── model_evaluation_metrics.csv # 評価指標出力（RMSEなど）<br>
├── plot_log_space.py # 結果の可視化スクリプト<br>
├── vif_result.csv # VIFの出力結果<br>
├── log_regression_standardized.py # 正規化重回帰分析実行<br>
├── log_regression_grouped_standardized_summary.txt　# 実行結果サマリ<br>
├── log_regression_grouped_standardized_result.csv # 実行結果CSV<br>
└── README.md # このファイル<br>
