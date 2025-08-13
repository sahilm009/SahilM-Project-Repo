# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:10:10 2021

@author: SMedepalli
"""
import pandas as pd
from fbprophet import Prophet
#from prophet.plot import plot_plotly, plot_components_plotly

df=pd.read_csv("C:\MLcodes\prophetDATA.csv")
df.head()
df.columns=['ds','y']


m=Prophet()
m.fit(df)
m=Prophet(daily_seasonality=True)

future=m.make_future_dataframe(periods=365)
future.tail()

forecast=m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

fig1=m.plot(forecast)
#fig2=m.plot_components(forecast)
