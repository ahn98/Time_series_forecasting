################ time step = 30min
################ row: 47,809

#%%
###############
## Data load ##
###############
import pandas as pd
import datetime
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import numpy as np
import missingno as msno
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
greenhouse_climate = pd.read_csv('/home/jy/3_dataset/AGIC/GreenhouseClimate_automato_modified.csv')
weather = pd.read_csv('/home/jy/3_dataset/AGIC/Weather_modified.csv')

'''
"gc" = greenhouse climate
"actu" = greenhouse actuator
"w" = outside weather
'''
gc = greenhouse_climate[['time','Tair','Rhair', 'CO2air']]
actu = greenhouse_climate[['time', 'VentLee', 'Ventwind', 'AssimLight', 'EnScr', 'PipeGrow', 'PipeLow', 'Tot_PAR']]
w = weather[['time', 'T_out', 'RH_out', 'I_glob', 'Winddir', 'Windsp']]

gc['time'] = pd.to_datetime(gc['time'])
actu['time'] = pd.to_datetime(actu['time'])
w['time'] = pd.to_datetime(w['time'])

###################
## Visualization ##
###################

#gc.plot.box(subplots = True, figsize = (50, 20))
#actu.plot.box(subplots =True, figsize = (50, 20))
#w.plot.box(subplots = True, figsize = (50, 20))
#msno.matrix(gc)  #값이 있으면 검은색, 값이 없으면 흰색으로 나타남
#msno.matrix(actu)
#msno.matrix(w)
#plt.show()


###############################
## Missing values & Outliers ##
###############################

gc2 = gc.fillna(method = 'backfill')
gc2 = gc2.fillna(method = 'ffill')

actu2 = actu.fillna(method = 'backfill')
actu2 = actu2.fillna(method = 'ffill')

w2 = w.fillna(method = 'backfill')
w2 = w2.fillna(method = 'ffill')

for c in gc2.columns:
    if gc2[c].dtype == float or gc2[c].dtype == int:
        q1 = gc2[c].quantile(0.25)
        q3 = gc2[c].quantile(0.75)
        IQR = q3 - q1
        gc2[c] = gc2[c].mask(gc2[c] < q1 - 1.5 * IQR, np.nan)
        gc2[c] = gc2[c].mask(gc2[c] > q3 + 1.5 * IQR, np.nan)
        print("Column : " + c + "\'s outliers which out of IQR are removed.")
gc2.reset_index(drop=True, inplace=True)

for c in actu2.columns:
    if actu2[c].dtype == float or actu2[c].dtype == int:
        q1 = actu2[c].quantile(0.25)
        q3 = actu2[c].quantile(0.75)
        IQR = q3 - q1
        actu2[c] = actu2[c].mask(actu2[c]< q1 - 1.5 * IQR, np.nan)
        actu2[c] = actu2[c].mask(actu2[c]> q3 + 1.5 * IQR, np.nan)
        print("Column : " + c + "\'s outliers which out of IQR are removed.")
actu2.reset_index(drop=True, inplace=True)

for c in w2.columns:
    if w2[c].dtype == float or w2[c].dtype == int:
        q1 = w2[c].quantile(0.25)
        q3 = w2[c].quantile(0.75)
        IQR = q3 - q1
        w2[c] = w2[c].mask(w2[c]< q1 - 1.5 * IQR, np.nan)
        w2[c] = w2[c].mask(w2[c]> q3 + 1.5 * IQR, np.nan)
        print("Column : " + c + "\'s outliers which out of IQR are removed.")
w2.reset_index(drop=True, inplace=True)


###########
## Merge ##
###########
gcw = pd.merge(gc2, w2, on = 'time', how = 'left') # gcw = gc+w 합침

gcw2 = gcw.fillna(method = 'ffill')
gcw3 = gcw2.fillna(method = 'backfill')

gcwactu = pd.merge(gcw3, actu2, on = 'time', how = 'left')  #gcwactu = gcw+actu 합침
gcwactu2 = gcwactu.fillna(method = 'ffill')
gcwactu3 = gcwactu2.fillna(method = 'backfill')

gcwactu3.set_index('time', inplace = True)

#%%
# 30분 간격으로 평균냄
gcwactu3_30min = gcwactu3.resample('30T').mean()

#%%
gcwactu3_30min.info()
#%%
#gcwactu3_30min.to_csv('paper_exp3.csv')
