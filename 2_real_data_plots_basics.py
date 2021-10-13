#%% functions

import statsmodels.api as sm
from statsmodels.api import OLS
import matplotlib.pyplot as plt

def ts_avg_yr_KR(Vec, LTG, n):
    # short run dynamics around formation
    N = LTG.shape[1] # time periods
    avg = np.zeros([8, N - 8])
    for t in range(3, N - 5):
        fhltg = np.nanpercentile(LTG[:, t], 90)
        flltg = np.nanpercentile(LTG[:, t], 10)
        if n == 1:
            for j in range(-3,5):
                avg[j + 3, t-3] = np.nanmedian(Vec[LTG[:, t] > fhltg, t + j])
                # print [t+j, Vec[1, t+j]]
        if n == 0:
            for j in range(-3, 5):
                avg[j + 3, t-3] = np.nanmedian(Vec[LTG[:, t] < flltg, t + j])
    avg = np.nanmean(avg[:, 3:N-8], axis = 1)
    return avg

def decile_returns_KR(LTG, Ret):
    N = LTG.shape[1] # time periods
    dec_returns = np.zeros([N-1, 10])
    for t in range(N-1):
        for i in range(10):
            fltg_low = np.nanpercentile(LTG[:, t], i * 10)
            fltg_high = np.nanpercentile(LTG[:, t], (i + 1) * 10)
            ltg = (LTG[:, t] > fltg_low) & (LTG[:, t] < fltg_high)
            if len(ltg)<1:
                print(i)
                print('STOP')
            #LTG_decile_returns[t, i] = Ret[ltg, t + 1].prod() ** (1.0 / len(ltg))
            dec_returns[t, i] = np.nanmean(Ret[ltg, t+1]) # compute YEARL AHEAD returns
    return np.nanmean(dec_returns, axis=0)

def decile_returns_prev_KR(LTG, Ret):
    N = LTG.shape[1] # time periods
    dec_returns = np.zeros([N-1, 10])
    for t in range(N-1):
        for i in range(10):
            fltg_low = np.nanpercentile(LTG[:, t], i * 10)
            fltg_high = np.nanpercentile(LTG[:, t], (i + 1) * 10)
            ltg = (LTG[:, t] > fltg_low) & (LTG[:, t] < fltg_high)
            if len(ltg)<1:
                print(i)
                print('STOP')
            #LTG_decile_returns[t, i] = Ret[ltg, t + 1].prod() ** (1.0 / len(ltg))
            dec_returns[t, i] = np.nanmean(Ret[ltg, t])
    return np.nanmean(dec_returns, axis=0)

def FE_KR(LTG, EPS, n, h):
    # computes annual forecast errors of LTG (based on DE or on RE)
    # n = 0: lltg
    # n = 1: hltg
    # h is horizon at which LTG is computed
    N = LTG.shape[1]
    fe = np.zeros(N - h)
    for t in range(1, N - h):
        fhltg = np.nanpercentile(LTG[:, t], 90)
        flltg = np.nanpercentile(LTG[:, t], 10)
        if n == 0:
            lltg = LTG[:, t] < flltg
            fe[t] = np.nanmedian((EPS[lltg, t + h]-EPS[lltg, t + h -1])/EPS[lltg, t + h -1] - LTG[lltg, t])
        if n == 1:
            hltg = LTG[:, t] > fhltg
            fe[t] = np.nanmedian((EPS[hltg, t + h]-EPS[hltg, t + h -1])/EPS[hltg, t + h -1]  - LTG[hltg, t])
    fe = np.nanmean(fe)
    return fe

def decile_KR(LTG, Vec):
    N = LTG.shape[1] # time periods
    dec_Vec = np.zeros([N-1, 10])
    for t in range(N-1):
        for i in range(10):
            fltg_low = np.nanpercentile(LTG[:, t], i * 10)
            fltg_high = np.nanpercentile(LTG[:, t], (i + 1) * 10)
            ltg = (LTG[:, t] > fltg_low) & (LTG[:, t] < fltg_high)
            if len(ltg)<1:
                print(i)
                print('STOP')
            dec_Vec[t, i] = np.nanmean(Vec[ltg, t])
    return np.nanmean(dec_Vec, axis=0)
#%% data reform
N = len(pd.unique(df.Ticker))
Ticker = np.reshape(np.array(df.Ticker), (N,-1))
Name = np.reshape(np.array(df.Name), (N,-1))
EPS = np.reshape(np.array(df.EPS), (N,-1))
Asset = np.reshape(np.array(df.Asset), (N,-1))
Sales = np.reshape(np.array(df.Sales), (N,-1))
Earnings = np.reshape(np.array(df.Earnings), (N,-1))
Equity = np.reshape(np.array(df.Equity), (N,-1))
Return = np.reshape(np.array(df.Return), (N,-1))
error_0 = np.reshape(np.array(df.error_0), (N,-1))
error_1 = np.reshape(np.array(df.error_1), (N,-1))
error_2 = np.reshape(np.array(df.error_2), (N,-1))

LTG_0 = np.reshape(np.array(df.LTG_0), (N,-1))
LTG_1 = np.reshape(np.array(df.LTG_1), (N,-1))
LTG_2 = np.reshape(np.array(df.LTG_2), (N,-1))

FY1 = np.reshape(np.array(df.FY1), (N,-1))
FY2 = np.reshape(np.array(df.FY2), (N,-1))
FY3 = np.reshape(np.array(df.FY3), (N,-1))
#%% Basic statistics  추가 데이터 받아서 마저 완성하기



EPS_decile = decile_KR(LTG_2, EPS)
Asset_decile = decile_KR(LTG_2, Asset)
Equity_decile = decile_KR(LTG_2, Equity)


plt.plot(EPS_decile)
plt.plot(Asset_decile)
plt.plot(Equity_decile)




#%% Figure 1. Decile portfolio returns (prev vs next)

dec_returns_1 = decile_returns_KR(LTG_2, Return)
dec_returns_0 = decile_returns_prev_KR(LTG_2, Return)
plt.plot(dec_returns_1)
plt.plot(dec_returns_0)

#%% Figure 2. Evolution of EPS

EPS_hltg = ts_avg_yr_KR(EPS, LTG_2, 1)[1:]
EPS_lltg = ts_avg_yr_KR(EPS, LTG_2, 0)[1:]

EPS_hltg = EPS_hltg/EPS_hltg[0]
EPS_lltg = EPS_lltg/EPS_lltg[0]

plt.plot(EPS_hltg)
plt.plot(EPS_lltg)

#%% Figure 3. Evolution of LTG

LTG_hltg = ts_avg_yr_KR(LTG_2, LTG_2, 1)[1:]
LTG_lltg = ts_avg_yr_KR(LTG_2, LTG_2, 0)[1:]

plt.plot(LTG_hltg)
plt.plot(LTG_lltg)

#%% Figure 4. Realized EPS versus LTG

HLTG_FE = np.zeros(3)
LLTG_FE = np.zeros(3)
lnEPS = np.log(EPS)

h = 0
LLTG_FE[h] = FE_KR(FY1/100, EPS, 0, h)  
HLTG_FE[h] = FE_KR(FY1/100, EPS, 1, h)

h = 1
LLTG_FE[h] = FE_KR(FY2/100, EPS, 0, h)  
HLTG_FE[h] = FE_KR(FY2/100, EPS, 1, h)

h = 2
LLTG_FE[h] = FE_KR(FY3/100, EPS, 0, h)  
HLTG_FE[h] = FE_KR(FY3/100, EPS, 1, h)
    
LLTG_FE = np.exp(LLTG_FE)-1
HLTG_FE = np.exp(HLTG_FE)-1    
plt.plot(HLTG_FE)
plt.plot(LLTG_FE)

#%% Return dynamics of HLTG and LLTG

HLTG_returns = ts_avg_yr_KR(Return, LTG_2, 1)
LLTG_returns = ts_avg_yr_KR(Return, LTG_2, 0)

plt.plot(HLTG_returns)
plt.plot(LLTG_returns)
