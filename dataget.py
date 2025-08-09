import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# terget:銘柄コード
class StockData:
    def __init__(self, target,period):
        self.target = target
        self.period = period
        self.ticker = yf.Ticker(self.target)
        self.df = None

    def get_stock_data(self):
        self.df = self.ticker.history(period=self.period)
        return self.df

class ComData:
    def __init__(self,period):
        self.period = period
        self.df = None
    
    def get_jpy(self):
        self.df = yf.Ticker("JPY=X").history(period=self.period)
        return self.df
    #エネルギー系 WTI原油先物
    def get_energy(self):
        self.df = yf.Ticker("CL=F").history(period=self.period)
        return self.df
    #金属系 金
    def get_metalX(self):
        self.df = yf.Ticker("GC=F").history(period=self.period)
        return self.df
    #農産物 小麦
    def get_agricultural_products(self):
        self.df = yf.Ticker("ZW=F").history(period=self.period)
        return self.df
    # 純利益
    def get_net_income(self):
        financials_quarter = self.ticker.quarterly_financials
        try:
            net_income = financials_quarter.loc['Net Income']
            return net_income
        except KeyError:
            return None

    def get_quarterly_total_revenue(self):
        financials = self.ticker.quarterly_financials
        try:
            total_revenue = financials.loc['Total Revenue']
            total_revenue.index = pd.to_datetime(total_revenue.index)
            return total_revenue.sort_index()
        except KeyError:
            print("Total Revenue not found in quarterly financials.")
            return None

    def get_quarterly_total_assets(self):
        balance_sheet = self.ticker.quarterly_balance_sheet
        try:
            total_assets = balance_sheet.loc['Total Assets']
            total_assets.index = pd.to_datetime(total_assets.index)
            return total_assets.sort_index()
        except KeyError:
            print("Total Assets not found in quarterly balance sheet.")
            return None

    def calc_total_asset_turnover(self):
        revenue = self.get_quarterly_total_revenue()
        assets = self.get_quarterly_total_assets()

        if revenue is None or assets is None:
            print("必要なデータが不足しています。")
            return None

        # 売上高と総資産の共通の日時インデックスで結合
        df = pd.DataFrame({
            'Total Revenue': revenue,
            'Total Assets': assets
        }).dropna()

        # 総資産回転率計算
        df['Total Asset Turnover'] = df['Total Revenue'] / df['Total Assets']

        return df


def test():
    stock = StockData("8306.T")  # 三菱UFJ
    #df = stock.get_stock_data("3y")
    df = stock.get_energy("3y")
    net_income = stock.get_net_income()
    
    print("株価データ（head）:\n", df.head())
    print("純利益四半期データ:\n", net_income)

    if net_income is not None:
        fig, ax1 = plt.subplots(figsize=(12,6))

        # 株価の線グラフ（左Y軸）
        ax1.plot(df.index, df["Close"], color="blue", label="Close Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        # 右Y軸を作成して純利益の棒グラフを描く
        ax2 = ax1.twinx()
        ax2.bar(net_income.index, net_income.values, width=20, color="red", alpha=0.6, label="Net Income")
        ax2.set_ylabel("Net Income", color="red")
        ax2.tick_params(axis='y', labelcolor='red')

        # グラフタイトルと凡例
        plt.title("Stock Close Price and Quarterly Net Income")
        fig = plt.gcf()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()
    else:
        print("純利益データが取得できませんでした。")
 
def test2():
    stock = StockData("9432.T")
    df = stock.get_stock_data("1y")
    total_asset_turnover_df = stock.calc_total_asset_turnover()  # 変数名変更

    print("株価データ（head）:\n", df.head())
    print("総資産回転率データ:\n", total_asset_turnover_df)

    if total_asset_turnover_df is not None and not total_asset_turnover_df.empty:
        fig, ax1 = plt.subplots(figsize=(12,6))

        # 株価の線グラフ（左Y軸）
        ax1.plot(df.index, df["Close"], color="blue", label="Close Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        # 右Y軸を作成して総資産回転率の棒グラフを描く
        ax2 = ax1.twinx()
        ax2.bar(total_asset_turnover_df.index, total_asset_turnover_df['Total Asset Turnover'], 
                width=20, color="red", alpha=0.6, label="Total Asset Turnover")
        ax2.set_ylabel("Total Asset Turnover", color="red")
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Stock Close Price and Total Asset Turnover")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()
    else:
        print("総資産回転率のデータがありません。")
        
def test3():
    stock = StockData("8306.T")  # 三菱UFJ
    df = stock.get_stock_data("3y")
    jpy = stock.get_jpy("3y")
    net_income = stock.get_net_income()
    
    print("株価データ（head）:\n", df.head())
    print("純利益四半期データ:\n", net_income)

    if net_income is not None:
        fig, ax1= plt.subplots(figsize=(12,6))

        # 株価の線グラフ（左Y軸）
        ax1.plot(df.index, df["Close"], color="blue", label="Close Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        # 右Y軸を作成して純利益の棒グラフを描く
        ax2 = ax1.twinx()  # 右Y軸
        ax2.plot(jpy.index, jpy["Close"], color="red", label="Close Price")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Close Price", color="red")
        ax2.tick_params(axis='y', labelcolor='red')

        # グラフタイトルと凡例
        plt.title("Stock Close Price and Quarterly Net Income")
        fig = plt.gcf()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.show()
    else:
        print("純利益データが取得できませんでした。")
 
if __name__ == "__main__":
	#単体テスト
	test3()