# get ready to start
all = [var for var in globals() if var[0] != "_"]
for var in all:
    del globals()[var]
import pandas as pd
import numpy as np
import os
os.chdir('G:/내 드라이브/대학원/대학원생활/2021-2 행태재무 develop/code')

#%% dataset for EPS and LTG
raw_company_KSE = pd.read_excel('dataset.xlsx',sheet_name='KSE_LTG')[7:9].T.iloc[1:].reset_index(drop = True)
raw_company_KSE.columns = ['Ticker', 'Name']
raw_company_KOSDAQ = pd.read_excel('dataset.xlsx',sheet_name='KOSDAQ_LTG')[7:9].T.iloc[1:].reset_index(drop = True)
raw_company_KOSDAQ.columns = ['Ticker', 'Name']
raw_company = pd.concat([raw_company_KSE, raw_company_KOSDAQ], axis = 0)
del [raw_company_KSE, raw_company_KOSDAQ]
raw_values_KSE = pd.read_excel('dataset.xlsx',sheet_name='KSE_LTG', header = 13)
raw_values_KOSDAQ = pd.read_excel('dataset.xlsx',sheet_name='KOSDAQ_LTG', header = 13)
raw_values_KOSDAQ = raw_values_KOSDAQ.drop(columns = 'Frequency')
raw_values = pd.concat([raw_values_KSE, raw_values_KOSDAQ], axis = 1)
del [raw_values_KSE, raw_values_KOSDAQ]
raw_values = raw_values.replace(['흑전','적전','흑지','적지'],[np.nan, np.nan, np.nan, np.nan])
raw_values = raw_values.replace('N/A(IFRS)',np.nan)
Ticker_temp = pd.unique(raw_company.Ticker)
Ticker = pd.DataFrame(Ticker_temp.repeat(raw_values.shape[0]))
Name_temp = pd.unique(raw_company.Name)
Name = pd.DataFrame(Name_temp.repeat(raw_values.shape[0]))
Time = pd.concat([raw_values.Frequency]*pd.unique(Ticker_temp).shape[0], ignore_index=True)
temp = np.mat(raw_values)[:,1:].astype('float')
i = 0
values = temp[:,(8*i):(8*i+8)]
for i in np.arange(1,(temp.shape[1]//8)):
    values_temp = temp[:,(8*i):(8*i+8)]
    values = np.concatenate((values, values_temp))    
values = pd.DataFrame(values, columns = ['EPS', 'FY1', 'FY2', 'FY3', 'Asset', 'Sales', 'Earnings', 'Equity'])
df = pd.concat([Ticker, Name, Time, values], axis = 1)
df.columns = ['Ticker', 'Name', 'Time', 'EPS', 'FY1', 'FY2', 'FY3', 'Asset', 'Sales', 'Earnings', 'Equity']
df.Time = df.Time.dt.strftime('%Y%m%d')
del [Ticker_temp, Ticker, Name, Name_temp, Time, temp, values, i, raw_company, raw_values]

#%% dataset for monthly returns
raw_company_KSE = pd.read_excel('dataset_returns.xlsx',sheet_name='KSE')[7:9].T.iloc[1:].reset_index(drop = True)
raw_company_KSE.columns = ['Ticker', 'Name']
raw_company_KOSDAQ = pd.read_excel('dataset_returns.xlsx',sheet_name='KOSDAQ')[7:9].T.iloc[1:].reset_index(drop = True)
raw_company_KOSDAQ.columns = ['Ticker', 'Name']
raw_company = pd.concat([raw_company_KSE, raw_company_KOSDAQ], axis = 0)
del [raw_company_KSE, raw_company_KOSDAQ]
raw_values_KSE = pd.read_excel('dataset_returns.xlsx',sheet_name='KSE', header = 13)
raw_values_KOSDAQ = pd.read_excel('dataset_returns.xlsx',sheet_name='KOSDAQ', header = 13)
raw_values_KOSDAQ = raw_values_KOSDAQ.drop(columns = 'Frequency')
raw_values = pd.concat([raw_values_KSE, raw_values_KOSDAQ], axis = 1)
del [raw_values_KSE, raw_values_KOSDAQ]
Ticker_temp = pd.unique(raw_company.Ticker)
Ticker = pd.DataFrame(Ticker_temp.repeat(raw_values.shape[0]))
Name_temp = pd.unique(raw_company.Name)
Name = pd.DataFrame(Name_temp.repeat(raw_values.shape[0]))
Time = pd.concat([raw_values.Frequency]*pd.unique(Ticker_temp).shape[0], ignore_index=True)
temp = np.mat(raw_values)[:,1:].astype('float')
i = 0
values = temp[:,(5*i):(5*i+5)]
for i in np.arange(1,(temp.shape[1]//5)):
    values_temp = temp[:,(5*i):(5*i+5)]
    values = np.concatenate((values, values_temp))    
values = pd.DataFrame(values, columns = ['open', 'close', 'return', 'corrected_open', 'return_div'])
df_returns = pd.concat([Ticker, Name, Time, values], axis = 1)
df_returns.columns = ['Ticker', 'Name', 'Time', 'open', 'close', 'Return', 'corrected_open', 'return_div']
df_returns.Time = df_returns.Time.dt.strftime('%Y%m%d')
del [Ticker_temp, Ticker, Name, Name_temp, Time, temp, values, values_temp, i, raw_company, raw_values]
df['Year'] = df.Time.str.slice(stop = 4)
df_returns['Year'] = df_returns.Time.str.slice(stop = 4)
df_returns = df_returns.groupby(['Ticker', 'Year']).sum()
df_returns.reset_index(inplace=True)  
df = pd.merge(df, df_returns, how = 'left', on = ['Ticker', 'Year'])
df['Return_plus_1'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['Return'].shift(-1)
del df_returns
