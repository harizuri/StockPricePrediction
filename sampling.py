import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# terget:銘柄コード
#銘柄別資源
class StockData:
    def __init__(self, target,period = "3y"):
        self.target = target
        self.period = period
        self.ticker = yf.Ticker(self.target)
        self.df = None
    
    def get_stock_data(self):
        self.df = self.ticker.history(period=self.period)
        return self.df
    
    def get_quarterly_total_revenue(self):
        financials = self.ticker.quarterly_financials
        try:
            total_revenue = financials.loc['Total Revenue']
            total_revenue.index = pd.to_datetime(total_revenue.index)
            return total_revenue.sort_index()
        except KeyError:
            print("TKeyError")
            return None
    
    def get_quarterly_total_assets(self):
        balance_sheet = self.ticker.quarterly_balance_sheet
        try:
            total_assets = balance_sheet.loc['Total Assets']
            total_assets.index = pd.to_datetime(total_assets.index)
            return total_assets.sort_index()
        except KeyError:
            print("KeyError")
            return None
    # 純利益
    def get_net_income(self):
        financials_quarter = self.ticker.quarterly_financials
        try:
            net_income = financials_quarter.loc['Net Income']
            return net_income
        except KeyError:
            print("KeyError")
            return None
    # Googleトレンド @@@
    def get_google_trend(self):
        pytrends = TrendReq(hl='ja-JP', tz=540)
        kw_list = ["キーワード"]  
        pytrends.build_payload(kw_list, timeframe='today 2025-07-30', geo='JP')
        # 時系列データ取得
        df = pytrends.interest_over_time()
        print(df)

# 共有資源
class ComData:
    def __init__(self,period = "3y"):
        self.period = period
        self.df = None
    # 
    def get_jpy(self):
        self.df = yf.Ticker("JPY=X").history(period=self.period)
        return self.df
    # エネルギー系 WTI原油先物
    def get_energy(self):
        self.df = yf.Ticker("CL=F").history(period=self.period)
        return self.df
    # 金属系 金
    def get_metalX(self):
        self.df = yf.Ticker("GC=F").history(period=self.period)
        return self.df
    # 農産物 小麦
    def get_agricultural_products(self):
        self.df = yf.Ticker("ZW=F").history(period=self.period)
        return self.df

#開発用
class yfinanceTool:
    def get_sector(self,ticker):
        ticker = yf.Ticker(ticker)
        info = ticker.info
        #print(info.get('sector'), info.get('industry'))
        return info.get('sector')

#データ数確認
class test_:
    def __init__(self, target,period = "3y"):
        self.target = target
        self.period = period
    
    def main(self):
        print("test_StockData")
        self.test_StockData()
        print("test_ComData")
        self.test_ComData()
    
    def test_StockData(self):
        self.run(StockData(self.target),StockData)
    
    def test_ComData(self):
        self.run(ComData(),ComData)
    
    def run(self,obj,Tclass):
        # __dict__ から関数を取り出して順に呼び出す
        for name, method in Tclass.__dict__.items():
            if callable(method) and not name.startswith("__"):
                try:
                    df = getattr(obj, name)()
                    print(name," : データ数", len(df))
                    #print(name," : データ数", count(df))
                except Exception as e:
                    print(f"Skipped {name}: {e}")


if __name__ == "__main__":
	#単体テスト
	#test = StockData("1605.T","3y")
	#test.get_google_trend()
	
	#test = yfinanceTool()
	#test.get_sector("1605.T")
	
	test = test_("1605.T")
	test.main()