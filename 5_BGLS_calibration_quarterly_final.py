#! /usr/bin/env python

#! #! /Library/Frameworks/Python.framework/Versions/2.7/bin/python

import numpy as np
import pandas as pd
#from BGLS_functions_final import ts_RE, ts_DE, ts_LTG

import os
os.chdir('G:/내 드라이브/대학원/2021-1 행태재무/행태 term paper')


#%% 0. Overview

# This file calibrates the model in Equations (1) and (2) using a method of simulated moments.
# It simulates Equations (1) and (2) for a range of parameter values.  
# For each parameter combination it computes the autocorrelations of log earnings (Section VI.B). 
# It also computes the diagnostic expectations of long term growth and the associated CG coefficients.
# It then computes the loss function defined in the text and stores the full results in a database.


#%% 1. Parameters 

# define parameter ranges (set narrowly to facilitate testing)
timescales = [1] 
r_a = np.arange(.01,.99,.1)
r_b = np.arange(.001,.999,.1)
r_sf = np.arange(.01,.5,.1)
r_se = np.arange(.01,.5,.1)
r_d = np.arange(0.01,3.,.1)
r_i = np.arange(1., 2., 1.)
r_H = [2]   
    # horizon for LTG in YEARS. Convention on other timescales: 
    # data generating process is in quarters; 
    # diagnostic expectations is in quarter multiples


# define matrix of parameters
L = [(H, a, b, sf, se, d, tscale, i) for H in r_H for a in r_a for b in r_b for sf in r_sf for se in r_se for d in r_d for tscale in timescales for i in r_i]
Lrho = []

# check size of matrix
count = 0
for (H, a, b, sf, se, d, tscale, i) in L:
    count = count + 1.
print(count)
count = 0


#%% 2. CG coefficients
# for each parameter combination:
# generate a long time series and compute compute the Coibion Gorodnichencko coefficient

for (H, a, b, sf, se, d, tscale, i) in L:

    paramlist = [H, a, b, sf, se, d, tscale, i]
    N = 50000                                # time periods
    f = np.zeros(N)                          # forcing term
    x = np.zeros(N)                          # log eps series
    eta = np.random.normal(0, sf, N)         # shocks to forcing term
    epsilon = np.random.normal(0, se, N)     # shocks to log eps

    # time series of fundamentals and observables
    for i in range(1,N):
        f[i] = a * f[i-1] + eta[i]
        x[i] = b * x[i-1] + f[i] + epsilon[i]

    # expectation time series
    B = sf ** 2 + (1 - a ** 2) * (se ** 2)
    su = ((- B + np.sqrt(B ** 2 + 4 * (a * sf * se) ** 2)) / (2 * a ** 2)) ** (
        1. / 2.)
    K = (a ** 2 * su ** 2 + sf ** 2) / (a ** 2 * su ** 2 + sf ** 2 + se ** 2)
    RE = ts_RE(a, b, K, x)
    DE = ts_DE(a, b, K, d, x, RE, tscale)
    LTG = ts_LTG(a, b, x, DE, 4 * H)
    LTGre = ts_LTG(a, b, x, RE, 4 * H)
    
    h = 4 * H    
    cond = - (1. - b ** h) + a ** h * (1. - (b / a) ** h) / (1. - (b / a)) * (K * (1+d) - a) / (1. - a)
    LTGyr = LTG[0 : : 4]
    xyr = x[0 : : 4]
    FE = xyr[8:len(xyr)] - xyr[4:len(xyr)-4] - LTGyr[4:len(xyr)-4]
    FR1 = LTGyr[4:len(xyr)-4] - LTGyr[3:len(xyr)-5]
    FR3 = LTGyr[4:len(xyr)-4] - LTGyr[1:len(xyr)-7]
    C3 = np.cov(FE,FR3)
    C1 = np.cov(FE,FR1)    
    CG3 = C3[0, 1] / C3[1, 1]
    CG1 = C1[0, 1] / C1[1, 1]
    paramlist.append(CG3)
    paramlist.append(CG1)
    
    for lag in range(1,5):
        C = np.cov(xyr[5:len(xyr)], xyr[5-lag:len(xyr)-lag]) 
        rho = C[0,1] / C[1,1]
        paramlist.append(rho)

    Lrho.append(paramlist)
    print(count)
    count = count + 1.

#%% 2. Loss function

cg_3yr = - 0.354
cg_1yr = - 0.378
#+ (rho3 - 0.409) ** 2 + (rho4 - 0.335) ** 2
def distance_corr(rho1,rho2,rho3,rho4,CG3, CG1):
    return np.sqrt((rho1 - 0.666) ** 2 + ( rho2 - 0.508) ** 2  + (CG3 - cg_3yr) ** 2 + (CG1 - cg_1yr) ** 2)

Calib = pd.DataFrame(Lrho, columns=['H','a', 'b', 'sf', 'se', 'd', 'tscale', 'i', 'CG3', 'CG1', 'rho1','rho2','rho3','rho4'])
Calib['distance_corr'] = Calib.apply(lambda x: distance_corr(x['rho1'], x['rho2'], x['rho3'], x['rho4'], x['CG3'], x['CG1']), axis=1)
Calib.to_csv('Calibration.csv', index = False)

# calibration result
H = 2
a = 0.91
b = 0.41
sf = 0.41
se = 0.31
d = 2.91
tscale = 1
i = 1.0
CG3 = -0.3118
CG1 = -0.3734
rho1 = 0.7179
rho2 = 0.4856
rho3 = 0.3281
rho4 = 0.2243

print((1+d)*K)
