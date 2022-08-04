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

graph = sns.scatterplot(x = "disk_id", y = 'model_y', palette = 'deep', hue='app', data = dfAllAppsSSDsLocation)
#hue_order=['WSM','none','RM','DB', 'WS', 'WPS', 'DAE'], 
#['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE']

graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')

plt.savefig('C11.png')
# %%

graph = sns.scatterplot(x = "disk_id", y = 'model_y', palette = 'deep', hue='app', data = dfAllAppsSSDsLocation, alpha=0.1)

graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')

plt.savefig('C12.png')

graph = sns.lineplot(data=dfAllAppsSSDsLocation, x="model_y", y="disk_id", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],estimator=np.mean, ci=90, markers=["o", "x", "+", "D", "v", "1", "s"])
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C13.png')


bin=np.arange(0,1088,25)
graph = sns.lmplot(x="disk_id", y="disk_id", hue="app", hue_order=['none','RM','WS','WSM', 'WPS', 'NAS', 'DB', 'SS', 'DAE'],data=dfAllAppsSSDsLocation, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)
graph.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.savefig('C112.png')