import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import pickle
import lightgbm as lgb
import sampling as DATA
import os

TEST_SPLIT_DATE = '2025-02-01'    # テストデータを使用する期間
PREDICATION_DATE = 14             # 何日後を予測するか"
MODEL_DIR = "../model/"

# terget:銘柄コード
class base_pattern:
    def __init__(self):
        self.stock_target = "8306.T"
        self.teain_period = "3y" #学習データの期間
        self.upward_rate = 0.07 # 何%の上昇を正解データにするか
        self.prediction_date = 14 #何日後を予測するか
        self.df_com = self.com_data(self.teain_period)
        self.feature_names = ""
    
    def prediction(self):
        print("****** prediction start ******")
        print("****** 予測 モデル読み込み ******")
        with open(MODEL_DIR+self.__class__.__name__+'_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("****** 予測データ サンプリング ******")
        stock = DATA.StockData(self.stock_target,"2wk")
        df = stock.get_stock_data()
        df['Weekday'] = df.index.weekday
        df.index = df.index.tz_localize(None)
        df = pd.get_dummies(df, columns=['Weekday'])
        df = df.merge(self.df_com[['JPY_Close','energy_Close','metalX_Close','ap_Close']], how='inner', left_index=True, right_index=True)
        latest = df.iloc[[-1]].copy() # 最新の特徴量1行だけ取得
        cols = ['Close', 'Volume', 'JPY_Close', 'energy_Close', 'metalX_Close', 'ap_Close'] \
       + [col for col in latest.columns if col.startswith('Weekday_')]
        latest = latest[cols]
        print("****** 予測 ******")
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0][1]
        print("****** 結果出力 ******")
        last_date = df.index[-1]
        two_weeks_later = last_date + pd.Timedelta(days=14)
        print("パターン名:",self.__class__.__name__)
        print(f"予測日: {two_weeks_later.strftime('%m 月 %d 日')}")
        print(f"予測結果: {'上昇 (買いシグナル)' if pred == 1 else '上昇せず'}")
        print(f"上昇確率: {prob:.2%}")
        print("****** end ******")
        return pred,prob
    
    def train_main(self):
        print("****** train start ******")
        print("****** 学習データ サンプリング ******")
        features, target, dates = self.make_data(self.stock_target,self.teain_period)
        print("****** 学習 ******")
        model, y_test, y_pred = self.train_and_evaluate(features, target)
        print("****** 学習 ログ作成 ******")
        importances = model.feature_importances_
        # DataFrameにまとめてソート
        feat_imp = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        print(feat_imp)
        print("****** モデル保存 ******")
        with open(MODEL_DIR+self.__class__.__name__+'_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            print("モデル保存:",MODEL_DIR+self.__class__.__name__+'_model.pkl')
        print("****** end ******")
        return model
        
    def train_and_evaluate(self, X, y):
        from sklearn.model_selection import train_test_split
        self.feature_names = X.columns
        # 学習とテストの分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        
        # アンダーサンプリング 正:負を合わせる
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
        print("アンダーサンプリング:", pd.Series(y_train_resampled).value_counts())
        
        # 学習とテスト
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        return model, y_test, y_pred
        
    def make_data(self,target,period):
        stock = DATA.StockData(target,period)
        # 正解データ
        df = stock.get_stock_data()
        # ラベル作成：2週間後に7%以上上昇するか
        df['Target'] = ((df['Close'].shift(-(self.prediction_date)) - df['Close']) / df['Close'] >= self.upward_rate).astype(int)
        print(target)
        #print(df['Target'].value_counts().get(1, 0)) # ターゲット割合、Trueが少なくなりすぎないように調整すること
        print(df['Target'].value_counts())
        df['Weekday'] = df.index.weekday
        df = pd.get_dummies(df, columns=['Weekday'])
        df.index = df.index.tz_localize(None)
        # 共有のデータを結合
        df = df.merge(self.df_com[['JPY_Close','energy_Close','metalX_Close','ap_Close']], how='inner', left_index=True, right_index=True)
        # 欠損処理
        df['JPY_Close'] = df['JPY_Close'].ffill()
        df.dropna(subset=['Target'], inplace=True)  # 特にターゲットのNaNを削除
        feature_cols = ['Close', 'Volume', 'JPY_Close','energy_Close','metalX_Close','ap_Close'] + [col for col in df.columns if col.startswith('Weekday_')]
        features = df[feature_cols]
        target = df['Target']
        print("最終データ数")
        print(df['Target'].value_counts())
        return features, target, df.index
        
    def com_data(self,period):
        stock = DATA.ComData(period)
        # 為替データ取得
        df_fx = stock.get_jpy()
        df_fx.index = df_fx.index.tz_localize(None)
        df = df_fx
        df.rename(columns={'Close': 'JPY_Close'}, inplace=True)
        # エネルギー系
        df_energy = stock.get_energy()
        df_energy.index = df_energy.index.tz_localize(None)
        df = df.merge(df_energy[['Close']], how='inner', left_index=True, right_index=True)
        df.rename(columns={'Close': 'energy_Close'}, inplace=True)
        # 金属系
        df_metalX = stock.get_metalX()
        df_metalX.index = df_metalX.index.tz_localize(None)
        df = df.merge(df_metalX[['Close']], how='inner', left_index=True, right_index=True)
        df.rename(columns={'Close': 'metalX_Close'}, inplace=True)
        # 農産物系
        df_ap = stock.get_agricultural_products()
        df_ap.index = df_ap.index.tz_localize(None)
        df = df.merge(df_ap[['Close']], how='inner', left_index=True, right_index=True)
        df.rename(columns={'Close': 'ap_Close'}, inplace=True)
        
        return df
        
if __name__ == "__main__":
    pattern = base_pattern()
    # 学習
    pattern.train_main()
    # 予測
    pattern.prediction()