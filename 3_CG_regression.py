

import statsmodels.api as sm
from statsmodels.api import OLS
import matplotlib.pyplot as plt

y_vars = np.array(['error_0', 'error_1', 'error_2'])
X_vars = np.mat([['delta_LTG_0_1', 'delta_LTG_1_1', 'delta_LTG_2_1'],
              ['delta_LTG_0_2', 'delta_LTG_1_2', 'delta_LTG_2_2'],
              ['delta_LTG_0_3', 'delta_LTG_1_3', 'delta_LTG_2_3']])


table_2_params = np.array([np.nan]*3*3).reshape(-1,3)
table_2_pvalues = np.array([np.nan]*3*3).reshape(-1,3)

for i in range(3): # h-th y variable
    for j in range(3): # k-th x variable with h-th y variable
        
        temp_y = y_vars[i]
        temp_X = [X_vars[i,j]] + time_dummies_names.tolist()

        model_temp = OLS(df[temp_y], sm.add_constant(df[temp_X]), missing = 'drop')
        res_temp = model_temp.fit()
        table_2_params[j,i] = res_temp.params[1]
        table_2_pvalues[j,i] = res_temp.pvalues[1]

del temp_X, temp_y, time_dummies_names, X_vars, y_vars, model_temp, res_temp, i, j

plt.scatter(df['error_0'], df['delta_LTG_0_3'])
plt.xlabel('LTG revision')
plt.ylabel('Forecast error')
plt.title('1. Prediction error of h=0 and k=3')


plt.scatter(df['error_1'], df['delta_LTG_1_3'])
plt.xlabel('LTG revision')
plt.ylabel('Forecast error')
plt.title('2. Prediction error of h=1 and k=3')

plt.scatter(df['error_2'], df['delta_LTG_2_3'])
plt.xlabel('LTG revision')
plt.ylabel('Forecast error')
plt.title('3. Prediction error of h=2 and k=3')

#%% autocorrelation

df_temp = df[['EPS', 'EPS_plus_1', 'EPS_plus_2', 'EPS_plus_3', 'EPS_plus_4']]
df_1 = df_temp[(df_temp.EPS>0) & (df_temp.EPS_plus_1>0)]
df_2 = df_temp[(df_temp.EPS>0) & (df_temp.EPS_plus_2>0)]
df_3 = df_temp[(df_temp.EPS>0) & (df_temp.EPS_plus_3>0)]
df_4 = df_temp[(df_temp.EPS>0) & (df_temp.EPS_plus_4>0)]

plt.scatter(np.log(df_1.EPS_plus_1), np.log(df_1.EPS))
plt.scatter(np.log(df_2.EPS_plus_2), np.log(df_2.EPS))
plt.scatter(np.log(df_3.EPS_plus_3), np.log(df_3.EPS))
plt.scatter(np.log(df_4.EPS_plus_4), np.log(df_4.EPS))

cov1 = np.cov(np.log(df_1.EPS_plus_1), np.log(df_1.EPS))
rho1 = cov1[0,1]/cov1[0,0]
cov2 = np.cov(np.log(df_2.EPS_plus_2), np.log(df_2.EPS))
rho2 = cov2[0,1]/cov2[0,0]
cov3 = np.cov(np.log(df_3.EPS_plus_3), np.log(df_3.EPS))
rho3 = cov3[0,1]/cov3[0,0]
cov4 = np.cov(np.log(df_4.EPS_plus_4), np.log(df_4.EPS))
rho4 = cov4[0,1]/cov4[0,0]
rhos = [rho1, rho2, rho3, rho4]

del df_1, df_2, df_3, df_4, cov1, cov2, cov3, cov4, rho1, rho2, rho3, rho4, df_temp
