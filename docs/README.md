# 仕様書
## 目的・概要
機械学習の学習と合わせて高品質な株価予測モデルを作成し、実用化を図る

## 基本使用
本システムは以下の機能を有する
・株価の予測モデルの学習および推論
・株価データの分析
・バッチ処理の起動
・デスクトップアプリケーションからの起動対応

## 予測モデル
### base_pattern
 Accuracy: 0.746
Classification Report:
              precision    recall  f1-score   support
           0       0.89      0.76      0.82        54
           1       0.48      0.71      0.57        17
    accuracy                           0.75        71
   macro avg       0.69      0.73      0.70        71
weighted avg       0.79      0.75      0.76        71