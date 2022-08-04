# %%
## Importing libraries

from datetime import datetime, date, timedelta
from IPython.display import display, clear_output
import time
from array import *
import numpy as np
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from pylab import savefig
import seaborn as sns
import pandas as pd
import csv
import json
import math
import datetime as dt
import pymongo as pym


# %%
myclient = pym.MongoClient("mongodb://localhost:27017/")
mydb = myclient["AllAppsSSDsLocation"]
mycol = mydb["all_apps_s_s_ds_location"]

myqueryAllAppsSSDsLocation = { "model_y": { "$eq": "C1" }}
myfieldsAllAppsSSDsLocation = {"disk_id":1, "model_y":1, "app":1, "_id":0}

mydocAllAppsSSDsLocation = mycol.find(myqueryAllAppsSSDsLocation, myfieldsAllAppsSSDsLocation)

# %%
listAllAppsSSDsLocation = list(mydocAllAppsSSDsLocation)
dfAllAppsSSDsLocation =  pd.DataFrame(listAllAppsSSDsLocation)

# %%
myclient = pym.MongoClient("mongodb://localhost:27017/")
mydb = myclient["SMARTAttributesFilterOverWear"]
mycol = mydb["s_m_a_r_t_att_over_wear"]

myquerySMARTAtt = { "model_x": { "$eq": "MC1" }}
myfieldsSMARTAtt = {"disk_id":1, "r_sectors":1, "w_l_count":1, "_id":0}

mydocSMARTAtt = mycol.find(myquerySMARTAtt, myfieldsSMARTAtt)

# %%
listSMARTAtt = list(mydocSMARTAtt)
dfSMARTAtt =  pd.DataFrame(listSMARTAtt)

# %%
dfSMARTAtt["model_y"] = "C1"

# %%
result = pd.merge(dfSMARTAtt, dfAllAppsSSDsLocation[['disk_id', 'model_y', 'app']], on=['disk_id', 'model_y'], how='inner')

# %%
#result.drop(result[result.w_r_d > 1500].index, inplace=True)
result.drop(result[result.r_sectors < 10].index, inplace=True)

maxWearValue = result["w_l_count"].max()

# %%
bin=np.arange(0,maxWearValue,150)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True, scatter=True, logx=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C18.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# %%
bin=np.arange(0,maxWearValue,100)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C19.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# %%
bin=np.arange(0,maxWearValue,50)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C110.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# %%
bin=np.arange(0,maxWearValue,25)    
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_ci="ci", ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C111.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# %%
bin=np.arange(0,maxWearValue,25)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C112.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# %%
bin=np.arange(0,maxWearValue,25)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.max, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C113.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# %%
bin=np.arange(0,maxWearValue,25)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app",hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'], data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=min, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C114.png')

plt.figure().clear()
plt.close()
plt.cla()
plt.clf()