#! /usr/bin/env python

#! #! /Library/Frameworks/Python.framework/Versions/2.7/bin/python
import numpy as np
import matplotlib.pyplot as plt
#from BGLS_functions_final import ts_eps, ts_RE, ts_DE, ts_LTG, ts_avg, ts_avg_yr, ts_price, distrib, FE, decile_returns
import os
os.chdir('G:/내 드라이브/대학원/2021-1 행태재무/행태 term paper')


#%% 0. Overview

# This file produces Figure 6 using the calibration of Table 3 (Section VI.B). 
# It first loads the calibration parameters and other relevant quantities.
# It then generates time series of fundamentals and of expectations for a large number of firms.
# It then computes prices and returns over time.
# And finaly plots the figures.


#%%  1. Fundamental parameters

a = 0.97              # persistence of forcing term
b = 0.56             # persistence of earnings
sf = 0.14           # var forcing term
se = 0.08           # var eps
d = 0.9              # theta
R = (1.+0.08)**0.25 # gross required return per quarter
tscale = 11          # time scale for DE is 12 quarters, 3 years.

# steady state signal to noise ratio
B = sf ** 2 + (1 - a ** 2) * (se ** 2)
su = ((- B + np.sqrt(B ** 2 + 4 * (a * sf * se) ** 2)) / (2 * a ** 2)) ** (1. / 2.)     # unconditional var in steady state
K = (a ** 2 * su ** 2 + sf ** 2) / (a ** 2 * su ** 2 + sf ** 2 + se ** 2)     # signal to noise ratio

# simulation parameters
F = 100             # firms (set small to facilitate testing)
N = 275             # time periods





CG3 = -0.3118
CG1 = -0.3734
rho1 = 0.7179
rho2 = 0.4856
rho3 = 0.3281
rho4 = 0.2243

#%%  2. Fundamentals and Expectations

# main variables
lnEPS = np.zeros([F,N])    # matrix of ln(eps) evolution
RE = np.zeros([F,N])       # matrix of Rational Expectations (RE) about current fundamental
DE = np.zeros([F,N])       # matrix of Diagnostic Expectations (DE) about current fundamental
LTG = np.zeros([F,N])      # matrix of Long Term Growth Expectations (LTG) 
LTGre = np.zeros([F,N])    # matrix of LTGs under RE

# time series of fundamentals and expectations for each firm
for j in range(0,F):
    lnEPS[j,:] = ts_eps(a, b, sf, se, N)     
    RE[j,:] = ts_RE(a, b, K, lnEPS[j,:])
    DE[j, :] = ts_DE(a, b, K, d, lnEPS[j, :], RE[j,:], tscale)
    LTG[j, :] = ts_LTG(a, b, lnEPS[j, :], DE[j, :], 16)    #  LTG is growth rate 4 YEARS ahead

# horizon to discount eps
S = 200                                 
V = np.zeros(S)
for s in range(S):
    V[s] = np.var(lnEPS[:,s])

# restrict sample to last N=200 periods for stability
N = 200
lnEPS = lnEPS[:,- N:]
RE = RE[:,- N:]
DE = DE[:,- N:]
LTG = LTG[:,- N:]

quarters = np.arange(N); years = quarters[3:N:4]

LTGyr = np.zeros([F, round(N/4)])
lnEPSyr = np.zeros([F, round(N/4)])
for j in range(0, F):
    LTGyr[j,:] = LTG[j,0::4]
    lnEPSyr[j,:] = lnEPS[j,0::4]

EPS_hltg_yr = ts_avg_yr(np.exp(lnEPSyr), LTGyr, 1)
EPS_lltg_yr = ts_avg_yr(np.exp(lnEPSyr), LTGyr, 0)
EPS_hltg_yr = EPS_hltg_yr/EPS_hltg_yr[0]
EPS_lltg_yr = EPS_lltg_yr/EPS_lltg_yr[0]


#%%  3. Compute Prices and Returns

PDE = np.zeros([F, N])     # matrix of prices under DE
RetDE = np.zeros([F, N])   # matrix of returns under DE
PRE  = np.zeros([F,N])     # matrix of prices under RE
RetRE  = np.zeros([F, N])  # matrix of returns under RE
RetAvg  = np.ones(N)
RetREyr = np.zeros([F, round(N/4)-1])
RetDEyr = np.zeros([F, round(N/4)-1])

for j in range(0, F):
    # time series for prices and returns
    PDE[j, :] = ts_price(a, b, lnEPS[j,:], DE[j, :], R, V)
    PRE[j, :] = ts_price(a, b, lnEPS[j,:], RE[j, :], R, V)
    for t in range(N):
        RetRE[j, t] = (np.exp(lnEPS[j, t]) + PRE[j, t]) / PRE[j, t - 1]
        RetDE[j, t] = (np.exp(lnEPS[j, t]) + PDE[j, t]) / PDE[j, t - 1]
        if (t % 4 == 0) & (t>0):
            RetREyr[j,round(t/4)-1] = np.prod(RetRE[j, t-3:t+1])
            RetDEyr[j,round(t/4)-1] = np.prod(RetDE[j, t-3:t+1])    
        
RetRE=RetRE[:, 2:N]
RetDE=RetDE[:, 2:N]

N = N-2
LTGyr = LTGyr[:,1:round(N/4)]

RetRE_hltg_yr = ts_avg_yr(RetREyr, LTGyr, 1)
RetRE_lltg_yr = ts_avg_yr(RetREyr, LTGyr, 0)
RetDE_hltg_yr = ts_avg_yr(RetDEyr, LTGyr, 1)
RetDE_lltg_yr = ts_avg_yr(RetDEyr, LTGyr, 0)


#%% 4. Figures


#%% Figure 6.1: year ahead return vs LTG

fig=plt.figure(figsize = (10,6))
LTG_decile_returns = decile_returns(LTGyr, RetDEyr)
ax1=fig.add_subplot(231)
ax1.plot(LTG_decile_returns, 'b')
ax1.set_xticks([0,3, 6, 9])
ax1.set_yticks([1.05, 1.1, 1.15])
ax1.set_yticklabels(['5%','10%','15%'])
ax1.set_xticklabels(['LLTG','4','7', 'HLTG'])
ax1.set_title('1. Annual returns vs LTG')


#%% Figure 6.2: EPS short run dynamics

EPS_hltg = ts_avg(np.exp(lnEPS), LTG, 1)
EPS_lltg = ts_avg(np.exp(lnEPS), LTG, 0)

EPS_hltg = EPS_hltg/EPS_hltg[0]
EPS_lltg = EPS_lltg/EPS_lltg[0]

ax2=fig.add_subplot(232)
ax2.plot(EPS_hltg,'r', label='HLTG')
ax2.plot(EPS_lltg,'b', label='LLTG')
legend = ax2.legend(loc='best')
ax2.set_xticks([4, 12, 20])
ax2.set_xticklabels(['-2','0','2'])
ax2.set_yticks([1,3,5])
ax2.set_yticklabels(['1','3','5'])
ax2.set_title('2. Evolution of EPS')


#%% Figure 6.3: LTG short run dynamics

LTG_hltg = ts_avg(LTG, LTG, 1)
LTG_lltg = ts_avg(LTG, LTG, 0)
ax3=fig.add_subplot(233)
ax3.plot(np.exp(0.25*LTG_hltg)-1,'r', label='HLTG')
ax3.plot(np.exp(0.25*LTG_lltg)-1,'b', label='LLTG')
ax3.set_xticks([4, 12, 20])
ax3.set_xticklabels(['-2','0','2'])
ax3.set_yticks([-0.2, 0. ,.2])
ax3.set_yticklabels(['-20%','0','20%'])
ax3.set_title('3. Evolution of LTG')


#%%
# Figure 6.4: forecast errors post formation

ltg = np.zeros([F,N])
ltg_re = np.zeros([F,N])
ltg_OC = np.zeros([F,N])
HLTG_FE = np.zeros(6)
LLTG_FE = np.zeros(6)
HLTG_FE_re = np.zeros(6)
LLTG_FE_re = np.zeros(6)
HLTG_FE_OC = np.zeros(6)
LLTG_FE_OC = np.zeros(6)
lnEPS = lnEPS[:,-N:]
LTG = LTG[:,-N:]
for h in range(1, 6):
    for j in range(F):   # for each firm, compute LTG at each point in time
        ltg[j, :] = ts_LTG(a, b, lnEPS[j, :], DE[j, :], h)
    LLTG_FE[h] = FE(ltg, lnEPS, 0, h)
    HLTG_FE[h] = FE(ltg, lnEPS, 1, h)
    
LLTG_FE = np.exp(0.25 * LLTG_FE)-1
HLTG_FE = np.exp(0.25 * HLTG_FE)-1
ax4=fig.add_subplot(234)
ax4.plot(HLTG_FE[1:6],'r', label='HLTG')
ax4.plot(LLTG_FE[1:6],'b', label='LLTG')
ax4.set_xticks([0,2,4])
ax4.set_xticklabels(['1','3','5'])
ax4.set_yticks([-0.2,0,0.2])
ax4.set_yticklabels(['-0.2','0','0.2'])
ax4.set_title('4. Forecast errors')


#%%
# Figure 6.5: Returns short run dynamics

ax5=fig.add_subplot(235)
ax5.plot(RetDE_hltg_yr,'r', label='HLTG')
ax5.plot(RetDE_lltg_yr,'b', label='LLTG')
ax5.set_yticks([.8, 1.2, 1.6])
ax5.set_yticklabels(['-20%','20%','60%'])
ax5.set_xticks([1, 3, 5])
ax5.set_xticklabels(['-2','0','2'])
ax5.set_title('5. Evolution of returns')


#%%
# Figure 6.6: kernel of truth

HLTG_growth, bins1 = distrib(lnEPS, LTG, 0)
nonHLTG_growth, bins2 = distrib(lnEPS, LTG, 1)
HLTG_forecast, bins3 = distrib(LTG, LTG, 4)
HLTG_growth = 1. * HLTG_growth / np.sum(HLTG_growth)
nonHLTG_growth = 1. * nonHLTG_growth / np.sum(nonHLTG_growth)
HLTG_forecast = 1. * HLTG_forecast / np.sum(HLTG_forecast)
ax6=fig.add_subplot(236)
ax6.plot(bins1[0:20], HLTG_growth, 'b', label = 'HLTG')
ax6.plot(bins2[0:20], nonHLTG_growth, 'g', label = 'nonHLTG')
line2 = np.sum(bins1[0:20] * HLTG_growth)
line3 = np.sum(bins3[0:20] * HLTG_forecast)
lim1 = 1.2 * np.max(HLTG_growth)
ax6.plot([line2, line2], [0., lim1], 'b--')
ax6.plot([line3, line3], [0., lim1], 'r', label = 'LTG')
ax6.set_yticks([])
ax6.set_title('6. Realized vs expected growth')
legend = ax6.legend(loc='best')

plt.subplots_adjust(wspace=0.3, hspace=0.4)

