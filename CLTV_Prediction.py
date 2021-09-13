
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


#reading dataset

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")

# Preprocessing the data
df.dropna(inplace=True)
df = df[~df['Invoice'].str.contains('C', na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

df['TotalPrice'] = df['Quantity'] * df['Price']
today_date = dt.datetime(2011, 12, 11)


# Preparing the data structure for lifetime prediction (RFM values)
cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                         lambda x: (today_date - x.min()).days],
                                         'Invoice': lambda x: x.nunique(),
                                         'TotalPrice': lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
# monetary should be the average value per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# frequency shoudl be more than 1
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
# recency and T should be shown as weekly
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df.head()

# Building BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 6-month expected number of purchases
cltv_df["expected_purc_24_week"] = bgf.predict(24,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

cltv_df.head()

plot_period_transactions(bgf)
plt.show()

#Building GG model

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

# Calculating CLTV by adding gg model
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time = 6,  # 6 month
                                   freq = "W",  # Weekly because we prepare the data as weekly
                                   discount_rate=0.01)

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)


cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)


# Creating a function for the whole process

def create_cltv_p(dataframe, month=6):
    # Preprocessing the data
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains('C', na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]

    dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['Price']
    today_date = dt.datetime(2011, 12, 11)

    replace_with_thresholds(df, "Quantity")
    replace_with_thresholds(df, "Price")

    # Preparing the data structure for lifetime prediction (RFM values)
    cltv_df = df.groupby('CustomerID').agg({'InvoiceDate': [lambda x: (x.max() - x.min()).days,
                                                        lambda x: (today_date - x.min()).days],
                                        'Invoice': lambda x: x.nunique(),
                                        'TotalPrice': lambda x: x.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    # monetary should be the average value per purchase
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    # frequency shoudl be more than 1
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    # recency and T should be shown as weekly
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    cltv_df.head()

    # Building BG/NBD model
    bgf = BetaGeoFitter(penalizer_coef=0.001)

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    # 6-month expected number of purchases
    cltv_df["expected_purc_24_week"] = bgf.predict(4*month,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
    # 1-week expected number of purchases
    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])
    # 1-month expected number of purchases
    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
    cltv_df.head()

    plot_period_transactions(bgf)
    plt.show()

    #Building GG model

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

    #  Calculating CLTV by adding gg model
    cltv = ggf.customer_lifetime_value(bgf,
                                     cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time = month,  # 6 month
                                    freq = "W",  # Weekly because we prepare the data as weekly
                                    discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv.sort_values(by="clv", ascending=False).head(50)


    cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

cltv_six = create_cltv_p(df,month=6)

cltv_six.head()

# Segmenting the customers based on CLTV
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()

cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)

cltv_final.groupby("segment").agg({"count", "mean", "sum"})

