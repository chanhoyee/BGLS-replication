#%% making LTG variables

# EPS shift and delta EPS, h = 0, 1, 2
df['EPS_minus_1'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['EPS'].shift(1)
df['EPS_plus_1'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['EPS'].shift(-1)
df['EPS_plus_2'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['EPS'].shift(-2)
df['EPS_plus_3'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['EPS'].shift(-3)
df['EPS_plus_4'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['EPS'].shift(-4)

df['delta_EPS_0'] = df.EPS/df.EPS_minus_1-1
df['delta_EPS_1'] = np.power(df.EPS_plus_1/df.EPS_minus_1, 1/2)-1
df['delta_EPS_2'] = np.power(df.EPS_plus_2/df.EPS_minus_1, 1/3)-1

# making LTG, h = 0, 1, 2

df['LTG_0'] = df['FY1']/100
df['LTG_1'] = (df['FY1']+df['FY2'])/2/100
df['LTG_2'] = (df['FY1']+df['FY2']+df['FY3'])/3/100

# LTG shift and delta LTG h = 0, 1, 2, k= 1, 2, 3

df['LTG_0_minus_1'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_0'].shift(1)
df['LTG_0_minus_2'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_0'].shift(2)
df['LTG_0_minus_3'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_0'].shift(3)

df['LTG_1_minus_1'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_1'].shift(1)
df['LTG_1_minus_2'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_1'].shift(2)
df['LTG_1_minus_3'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_1'].shift(3)

df['LTG_2_minus_1'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_2'].shift(1)
df['LTG_2_minus_2'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_2'].shift(2)
df['LTG_2_minus_3'] = df.sort_values(by=['Ticker','Time']).groupby('Ticker')['LTG_2'].shift(3)

df['delta_LTG_0_1'] = df['LTG_0'] - df['LTG_0_minus_1']
df['delta_LTG_0_2'] = df['LTG_0'] - df['LTG_0_minus_2']
df['delta_LTG_0_3'] = df['LTG_0'] - df['LTG_0_minus_3']

df['delta_LTG_1_1'] = df['LTG_1'] - df['LTG_0_minus_1']
df['delta_LTG_1_2'] = df['LTG_1'] - df['LTG_0_minus_2']
df['delta_LTG_1_3'] = df['LTG_1'] - df['LTG_0_minus_3']

df['delta_LTG_2_1'] = df['LTG_2'] - df['LTG_0_minus_1']
df['delta_LTG_2_2'] = df['LTG_2'] - df['LTG_0_minus_2']
df['delta_LTG_2_3'] = df['LTG_2'] - df['LTG_0_minus_3']

# making error variable for y

df['error_0'] = df['delta_EPS_0'] - df['LTG_0']
df['error_1'] = df['delta_EPS_1'] - df['LTG_1']
df['error_2'] = df['delta_EPS_2'] - df['LTG_2']

# time dummy

df_temp_time = pd.get_dummies(df.Time, drop_first = True)
time_dummies_names = np.array(df_temp_time.columns)
df_temp_time.columns = time_dummies_names
df_temp = df
df = pd.concat([df_temp, df_temp_time], axis = 1)

df_temp = df.replace([np.inf, -np.inf], np.nan).dropna()
table1 = df_temp.describe()

del df_temp, df_temp_time





