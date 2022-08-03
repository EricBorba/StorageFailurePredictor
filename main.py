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

myqueryAllAppsSSDsLocation = { "model_x": { "$eq": "MC1" }}
myfieldsAllAppsSSDsLocation = {"disk_id":1, "r_sectors":1, "w_l_count":1, "_id":0}

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

# %%
graph = sns.scatterplot(x = "w_l_count", y = 'r_sectors', palette = 'deep', hue='app', data = result)
#hue_order=['WSM','none','RM','DB', 'WS', 'WPS', 'DAE'], 
#['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE']

graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')

plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C11.pdf', dpi=300)

# %%
graph = sns.scatterplot(x = "w_l_count", y = 'r_sectors', palette = 'deep', hue='app', hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'], data = result, style="app", alpha=0.1)

graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')

plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C12.pdf', dpi=300)

# %%
graph = sns.scatterplot(x = "w_l_count", y = 'r_sectors', size="app", sizes=(1, 1), palette = 'deep', hue='app', hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data = result, style="app", alpha=0.1)

graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')

plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C13.pdf', dpi=300)

# %%
graph = sns.lineplot(data=result, x="w_l_count", y="r_sectors", hue="app",hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'], estimator=np.mean, ci=90, markers=True, err_style="bars")
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C14.pdf', dpi=300)

# %%
graph = sns.lineplot(data=result, x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],estimator=np.mean, ci=90, markers=["o", "x", "+", "D", "v", "1", "s"])
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C15.pdf', dpi=300)

# %%
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], palette="Set1", x_estimator=np.mean, x_ci="ci", ci=60, fit_reg=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C16.pdf', dpi=300)

# %%
bin=np.arange(0,1088,150) #[150,300,450,600,750,900,1050,1200,1350]
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C17.pdf', dpi=300)

# %%
bin=np.arange(0,1088,150)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True, scatter=True, logx=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C18.pdf', dpi=300)

# %%
bin=np.arange(0,1088,100)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C19.pdf', dpi=300)

# %%
bin=np.arange(0,1088,50)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", ci=90, fit_reg=True, x_bins=bin, truncate=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C110.pdf', dpi=300)

# %%
bin=np.arange(0,1088,25)    
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_ci="ci", ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C111.pdf', dpi=300)

# %%
bin=np.arange(0,1088,25)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C112.pdf', dpi=300)

# %%
bin=np.arange(0,1088,25)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.max, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C113.pdf', dpi=300)

# %%
bin=np.arange(0,1088,25)
graph = sns.lmplot(x="w_l_count", y="r_sectors", hue="app",hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'], data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=min, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('/home/rodrigue/myenv/analysis/WLxRS/C1Model/C114.pdf', dpi=300)