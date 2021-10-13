#! /usr/bin/env python

#! #! /Library/Frameworks/Python.framework/Versions/2.7/bin/python

import numpy as np


#ts_eps, ts_RE, ts_DE, ts_LTG, ts_avg, ts_avg_yr, ts_price, distrib, FE, decile_returns

def ts_eps(a, b, sf, se, N):
    # time series of ln(eps)
    f = np.zeros(N)  # forcing term
    x = np.zeros(N)  # log eps
    eta = np.random.normal(0, sf, N)  # shocks to forcing term
    epsilon = np.random.normal(0, se, N)  # shocks to log eps
    f0 = 0.
    x[0] = 10
    for i in range(1,N):
        f[i] = (1 - a) * f0 + a * f[i-1] + eta[i]     # forcing term
        x[i] = b * x[i-1] + f[i] + epsilon[i]         # log eps
    return x


def ts_RE(a, b, K, x):
    # Rational Expectations of CURRENT forcing term
    N = len(x)
    RE = np.zeros(N)
    for i in range(1, N):
        RE[i] = a * RE[i - 1] + K * (x[i] - b * x[i-1] - a * RE[i - 1])
    return RE


def ts_DE(a, b, K, d, x, RE, tscale):
    # Diagnostic Expectations of CURRENT forcing term
    N = len(x)
    DE = np.zeros(N)
    for i in range(tscale, N):
        DE[i] = RE[i] + d * K * (x[i] - b * x[i-1] - a ** tscale * RE[i - tscale])
    return DE


def ts_LTG(a, b, x, E, h):
    # LTG expectations under E = RE or DE
    # h is horizon
    N = len(x)
    LTG = np.zeros(N)
    for i in range(1, N):
        LTG[i] = - (1. - b ** h) * x[i] + a ** h * E[i] * (1. - (b / a) ** h) / (1. - (b / a))
    return LTG


def ts_avg(Vec, LTG, n):
    # short run dynamics around formation
    N = LTG.shape[1]
    avg = np.zeros([25, N - 13])
    for t in range(12, N - 13):
        fhltg = np.percentile(LTG[:, t], 90)
        flltg = np.percentile(LTG[:, t], 10)
        if n == 1:
            for j in range(-12,13):
                avg[j + 12, t] = Vec[LTG[:, t] > fhltg, t + j].mean()
                # print [t+j, Vec[1, t+j]]
        if n == 0:
            for j in range(-12, 13):
                avg[j + 12, t] = Vec[LTG[:, t] < flltg, t + j].mean()
    avg = np.mean(avg[:, 12:N-13], axis = 1)
    return avg

def ts_avg_yr(Vec, LTG, n):
    # short run dynamics around formation
    N = LTG.shape[1]
    avg = np.zeros([8, N - 8])
    for t in range(3, N - 5):
        fhltg = np.percentile(LTG[:, t], 90)
        flltg = np.percentile(LTG[:, t], 10)
        if n == 1:
            for j in range(-3,5):
                avg[j + 3, t-3] = Vec[LTG[:, t] > fhltg, t + j].mean()
                # print [t+j, Vec[1, t+j]]
        if n == 0:
            for j in range(-3, 5):
                avg[j + 3, t-3] = Vec[LTG[:, t] < flltg, t + j].mean()
    avg = np.mean(avg[:, 3:N-8], axis = 1)
    return avg



def ts_price(a, b, EPS, E, R, V):
    # time series of price for a firm
    # E is time series of expectations of fundamentals, R is return, V is vol of future eps
    N = E.shape[0]
    S = V.shape[0]
    Pr = 190*np.ones(N)
    for t in range(1, N):  # time periods
        ExpEPS = np.ones(S)
        Q = np.zeros(S)
        for s in range(1,S):
            Q[s] = a ** s * (1 - (b / a) ** s) / (1 - (b / a))
            ExpEPS[s] = np.exp((b ** s) * EPS[t] + Q[s] * E[t] + 0.5 * V[s]) / (R ** s)
        Pr[t] = np.sum(ExpEPS[1:S])
    return Pr


def decile_returns(LTG, Ret):
    N = LTG.shape[1]
    dec_returns = np.zeros([N-1, 10])
    for t in range(N-1):
        for i in range(10):
            fltg_low = np.percentile(LTG[:, t], i * 10)
            fltg_high = np.percentile(LTG[:, t], (i + 1) * 10)
            ltg = (LTG[:, t] > fltg_low) & (LTG[:, t] < fltg_high)
            if len(ltg)<1:
                print(i)
                print('STOP')
            #LTG_decile_returns[t, i] = Ret[ltg, t + 1].prod() ** (1.0 / len(ltg))
            dec_returns[t, i] = Ret[ltg, t + 1].mean()   # compute YEARL AHEAD returns
    return np.mean(dec_returns, axis=0)
    #return LTG_decile_returns[3:N - 4, :].prod(axis=0) ** (1.0 / (N - 7))


def FE(LTG, EPS, n, h):
    # computes forecast errors of LTG (based on DE or on RE)
    # n = 0: lltg
    # n = 1: hltg
    # h is horizon at which LTG is computed
    N = LTG.shape[1]
    fe = np.zeros(N - h)
    for t in range(1, N - h):
        fhltg = np.percentile(LTG[:, t], 90)
        flltg = np.percentile(LTG[:, t], 10)
        if n == 0:
            lltg = LTG[:, t] < flltg
            fe[t] = np.mean(EPS[lltg, t + h] - EPS[lltg, t] - LTG[lltg, t])
        if n == 1:
            hltg = LTG[:, t] > fhltg
            fe[t] = np.mean(EPS[hltg, t + h] - EPS[hltg, t] - LTG[hltg, t])
    fe = np.mean(fe)
    return fe


def distrib(Vec, LTG, n):
    # computes distributions of Vec conditional on LTG percentiles
    # n=0...3: distributions of future growth of Vec
    # n=4: current distribution of Vec
    N = LTG.shape[1]
    dist = []
    for i in range(1,N-16):
        f_hltg = np.percentile(LTG[:, i], 90)
        f_lltg = np.percentile(LTG[:, i], 10)
        if n == 0:                                  # HLTG
            dist.extend(0.25 * (Vec[LTG[:, i] > f_hltg, i + 16] - Vec[LTG[:, i] > f_hltg, i]))
        if n == 1:                                  # not HLTG
            dist.extend(0.25 * (Vec[LTG[:, i] < f_hltg, i + 16] - Vec[LTG[:, i] < f_hltg, i]))
        if n == 2:                                  # LLTG
            dist.extend(0.25 * (Vec[LTG[:, i] < f_lltg, i + 20] - Vec[LTG[:, i] < f_lltg, i]))
        if n == 3:                                  # not LLTG
            dist.extend(0.25 * (Vec[LTG[:, i] > f_lltg, i + 20] - Vec[LTG[:, i] > f_lltg, i]))
        if n == 4:
            dist.extend(0.25 * (Vec[LTG[:, i] > f_hltg, i]))    
    dist = np.exp(dist)     #  transform changes in lnEPS into EPS growth rates 
    hist, bin_edges = np.histogram(dist, bins = np.arange(0,2.1,0.1))
    return hist, bin_edges

