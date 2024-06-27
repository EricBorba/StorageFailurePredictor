#!/usr/bin/env python
# coding: utf-8

# In[133]:


## reset specific variables (replace regular_expression by the variables of interest)
#%reset_selective <regular_expression>

# reset all variables
get_ipython().run_line_magic('reset', '-f')


# In[1]:


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
from mongoengine import *
import statsmodels
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Creating/Connecting Mongo DB instances

# Provide the mongodb atlas url to connect python to mongodb using pymongo
#CONNECTION_STRING = "mongodb+srv://<jgu>:<123>@<cluster-jgu>.mongodb.net/SMARTAttributesFilter"

connect(db='SMARTAttributesFilter', alias='SMARTAttributesFilter_alias')

connect(db='SMARTAttributesFilterOverWear', alias='SMARTAttributesFilterOverWear_alias')

connect(db='OverTimeSSDsFailures', alias='OverTimeSSDsFailures_alias')

connect(db='AllAppsSSDsFailures', alias='AllAppsSSDsFailures_alias')

connect(db='AllAppsSSDsLocation', alias='AllAppsSSDsLocation_alias')

connect(db='AllDiskIDSMARTAttributes', alias='AllDiskIDSMARTAttributes_alias')

connect(db='AllDiskIDSMARTAttributesFirstDay', alias='AllDiskIDSMARTAttributesFirstDay_alias')

connect(db='FailuresAppsLocation', alias='FailuresAppsLocation_alias')

connect(db='SMARTAtt_FailuresAppsLocation', alias='SMARTAtt_FailuresAppsLocation_alias')


# In[3]:


## Setting document schema

# SMART attributes and disk information from the 2 years daily sampling dataset (500k disks)
class SMARTAtt(Document):
     disk_id = FloatField(required=False, default='0')
     timestamp = DateTimeField(required=False, default='0')
     model_x = StringField(required=False, default='0')
     r_sectors = FloatField(required=False, default='0')
     u_errors = FloatField(required=False, default='0')
     p_failedA = FloatField(required=False, default='0')
     p_failedB = FloatField(required=False, default='0')
     e_failedA = FloatField(required=False, default='0')
     e_failedB = FloatField(required=False, default='0')
     n_b_written = FloatField(required=False, default='0')
     n_b_read = FloatField(required=False, default='0')
     meta = {'db_alias': 'SMARTAttributesFilter_alias'}

# SMART attributes (related to wear) and disk information from the 2 years daily sampling dataset (500k disks)
class SMARTAttOverWear(Document):
     disk_id = FloatField(required=False, default='0')
     timestamp = DateTimeField(required=False, default='0')
     model_x = StringField(required=False, default='0')
     r_sectors = FloatField(required=False, default='0')
     w_l_count = FloatField(required=False, default='0')
     w_r_d = FloatField(required=False, default='0')
     media_wearout_i = FloatField(required=False, default='0')
     meta = {'db_alias': 'SMARTAttributesFilterOverWear_alias'}


# Failure time and disk information from the 2 years daily sampling dataset (500k disks)
class OverTimeSSDsFailures(Document):
     disk_id = FloatField(required=False, default='0')
     failure_time = DateTimeField(required=False, default='0')
     model_x = StringField(required=False, default='0')
     meta = {'db_alias': 'OverTimeSSDsFailures_alias'}

# Failure time and disk information (without model) from the full datset (1M disks)
class AllAppsSSDsFailures(Document):
     disk_id = FloatField(required=False, default='0')
     failure_time = DateTimeField(required=False, default='0')
     app = StringField(required=False, default='0')
     node_id = FloatField(required=False, default='0')
     rack_id = FloatField(required=False, default='0')
     machine_room_id = FloatField(required=False, default='0')
     meta = {'db_alias': 'AllAppsSSDsFailures_alias'}

# Apps and disks characteristics from the full dataset (1M disks)
class AllAppsSSDsLocation(Document):
     disk_id = FloatField(required=False, default='0')
     model_y = StringField(required=False, default='0')
     app = StringField(required=False, default='0')
     node_id = FloatField(required=False, default='0')
     rack_id = FloatField(required=False, default='0')
     slot_id = FloatField(required=False, default='0')
     meta = {'db_alias': 'AllAppsSSDsLocation_alias'}

# SMART attributes from the full dataset (1M disks)
class AllDiskIDSMARTAttributes(Document):
     disk_id = FloatField(required=False, default='0')
     model_y = StringField(required=False, default='0')
     r_sectors = FloatField(required=False, default='0')
     u_errors = FloatField(required=False, default='0')
     p_on = FloatField(required=False, default='0')
     p_c_count = FloatField(required=False, default='0')
     p_failedA = FloatField(required=False, default='0')
     p_failedB = FloatField(required=False, default='0')
     e_failedA = FloatField(required=False, default='0')
     e_failedB = FloatField(required=False, default='0')
     n_b_written = FloatField(required=False, default='0')
     n_b_read = FloatField(required=False, default='0')
     w_l_count = FloatField(required=False, default='0')
     w_r_d = FloatField(required=False, default='0')
     media_wearout_i = FloatField(required=False, default='0')
     meta = {'db_alias': 'AllDiskIDSMARTAttributes_alias'}

# First Day SMART attributes from the full dataset (1M disks)
class AllDiskIDSMARTAttributesFirstDay(Document):
     disk_id = FloatField(required=False, default='0')
     model_y = StringField(required=False, default='0')
     r_sectors = FloatField(required=False, default='0')
     u_errors = FloatField(required=False, default='0')
     p_on = FloatField(required=False, default='0')
     p_c_count = FloatField(required=False, default='0')
     p_failedA = FloatField(required=False, default='0')
     p_failedB = FloatField(required=False, default='0')
     e_failedA = FloatField(required=False, default='0')
     e_failedB = FloatField(required=False, default='0')
     n_b_written = FloatField(required=False, default='0')
     n_b_read = FloatField(required=False, default='0')
     w_l_count = FloatField(required=False, default='0')
     w_r_d = FloatField(required=False, default='0')
     media_wearout_i = FloatField(required=False, default='0')
     meta = {'db_alias': 'AllDiskIDSMARTAttributesFirstDay_alias'}

# Merge of OverTimeSSDsFailures and AllAppsSSDsFailures documents
class FailuresAppsLocation(Document):
     disk_id = FloatField(required=False, default='0')
     failure_time = DateTimeField(required=False, default='0')
     model_x = StringField(required=False, default='0')
     model_y = StringField(required=False, default='0')
     app = StringField(required=False, default='0')
     node_id = FloatField(required=False, default='0')
     rack_id = FloatField(required=False, default='0')
     machine_room_id = FloatField(required=False, default='0')
     meta = {'db_alias': 'FailuresAppsLocation_alias'}

class SMARTAtt_FailuresAppsLocation(Document):
     smart_att = ReferenceField(SMARTAtt)
     failures_app_location = ReferenceField(FailuresAppsLocation)
     meta = {'db_alias': 'SMARTAtt_FailuresAppsLocation_alias'}


# In[ ]:


## Closing the connection to the DB

disconnect(alias='SMARTAttributesFilter_alias')

disconnect(alias='SMARTAttributesFilterOverWear_alias')

disconnect(alias='OverTimeSSDsFailures_alias')

disconnect(alias='AllAppsSSDsFailures_alias')

disconnect(alias='AllAppsSSDsLocation_alias')

disconnect(alias='AllDiskIDSMARTAttributes_alias')

disconnect(alias='FailuresAppsLocation_alias')

disconnect(alias='SMARTAtt_FailuresAppsLocation_alias')


# In[60]:


## Variables to connect to the DB

myclient = pym.MongoClient("mongodb://localhost:27017/")
mydb = myclient["AllAppsSSDsLocation"]
mycol = mydb["all_apps_s_s_ds_location"]

myqueryAllAppsSSDsLocation = { "model_y": { "$eq": "C2" }}
myfieldsAllAppsSSDsLocation = {"disk_id":1, "model_y":1, "app":1, "_id":0}

mydocAllAppsSSDsLocation = mycol.find(myqueryAllAppsSSDsLocation, myfieldsAllAppsSSDsLocation)


myclient = pym.MongoClient("mongodb://localhost:27017/")
mydb = myclient["SMARTAttributesFilterOverWear"]
mycol = mydb["s_m_a_r_t_att_over_wear"]

myquerySMARTAtt = { "model_x": { "$eq": "MC2" }}
myfieldsSMARTAtt = {"disk_id":1, "r_sectors":1, "w_l_count":1, "_id":0}

mydocSMARTAtt = mycol.find(myquerySMARTAtt, myfieldsSMARTAtt)


# In[ ]:


### Code for plotting using chuncks (not currently used)

# Create an axis for both DataFrames to be plotted on
fig, ax = plt.subplots()

slices=2
chunk=57637063
list2 =[]
df2 = pd.DataFrame()
bin=np.arange(0,6500,25)
for i in range(1,(slices+1)):
    list2 = list(mycol.find(myquery, myfields)[(chunk*(i-1)):(chunk*i)])
    df2 = pd.DataFrame(list2)
    df2.drop(df2[df2.r_sectors < 10].index, inplace=True)
    df2["model_y"] = "C1"
    result = pd.merge(df2, df_AlibabaSnapShot_AllAppsSSDsLocation[['disk_id', 'model_y', 'app']], on=['disk_id', 'model_y'], how='inner')
    if i == slices:
        ax = sns.lmplot(x="w_l_count", y="r_sectors", hue="app",data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True, legend=False)        
    else:
        ax = sns.lmplot(x="w_l_count", y="r_sectors", hue="app",data=result, palette="Set1", markers=["o", "x", "+", "D", "v", "1", "s", "<", ">"], x_estimator=np.mean, x_ci="ci", x_bins=bin, ci=None, fit_reg=False, truncate=True, scatter=True)        
    del list2, result, df2
    gc.collect()

ax.set(xlabel ="wear leveling", ylabel = "reallocated sectors", title ='C1 SSD Model')
plt.show()
plt.savefig('/root/analysisJul29/C1Model/C112.pdf', dpi=300)

    


# In[ ]:


# Querying/Reading the OverTimeSSDsFailures database (mongodb) and turning It into a pandas dataframe

doc_AlibabaOver_Failurelogs = OverTimeSSDsFailures.objects()

jSon_AlibabaOver_Failurelogs = json.loads(doc_AlibabaOver_Failurelogs.to_json())
df_AlibabaOver_Failurelogs = pd.DataFrame.from_dict(jSon_AlibabaOver_Failurelogs) 

# Modifying the mongo db date type to some human-readable type
dicDateToString = json.dumps(list(df_AlibabaOver_Failurelogs['failure_time']))
dicStringToJson = json.loads(dicDateToString)
dicJsonToDf = pd.DataFrame.from_dict(dicStringToJson)
df_AlibabaOver_Failurelogs['failure_time'] = dicJsonToDf['$date']


# In[ ]:


# Querying/Reading the AllAppsSSDsFailures database (mongodb) and turning It into a pandas dataframe

doc_AlibabaSnapShot_FailuresAppsLocation = AllAppsSSDsFailures.objects()

jSon_AlibabaSnapShot_FailuresAppsLocation = json.loads(doc_AlibabaSnapShot_FailuresAppsLocation.to_json())
df_AlibabaSnapShot_FailuresAppsLocation = pd.DataFrame.from_dict(jSon_AlibabaSnapShot_FailuresAppsLocation) 

# Modifying the mongo db date type to some human-readable type
dicDateToString = json.dumps(list(df_AlibabaSnapShot_FailuresAppsLocation['failure_time']))
dicStringToJson = json.loads(dicDateToString)
dicJsonToDf = pd.DataFrame.from_dict(dicStringToJson)
df_AlibabaSnapShot_FailuresAppsLocation['failure_time'] = dicJsonToDf['$date']


# In[4]:


# Querying/Reading the AllAppsSSDsLocation database (mongodb)

doc_AlibabaSnapShot_AllAppsSSDsLocation = AllAppsSSDsLocation.objects()

jSon_AlibabaSnapShot_AllAppsSSDsLocation = json.loads(doc_AlibabaSnapShot_AllAppsSSDsLocation.to_json())
df_AlibabaSnapShot_AllAppsSSDsLocation = pd.DataFrame.from_dict(jSon_AlibabaSnapShot_AllAppsSSDsLocation) 


# In[21]:


# Querying/Reading the AllDiskIDSMARTAttributes database (mongodb)

doc_AlibabaSnapShot_AllDiskIDSMARTAttributes = AllDiskIDSMARTAttributes.objects()

jSon_AlibabaSnapShot_AllDiskIDSMARTAttributes = json.loads(doc_AlibabaSnapShot_AllDiskIDSMARTAttributes.to_json())
df_AlibabaSnapShot_AllDiskIDSMARTAttributes = pd.DataFrame.from_dict(jSon_AlibabaSnapShot_AllDiskIDSMARTAttributes) 


# In[22]:


# Querying/Reading the AllDiskIDSMARTAttributesFirstDay database (mongodb)

doc_AlibabaSnapShot_AllDiskIDSMARTAttributesFirstDay = AllDiskIDSMARTAttributesFirstDay.objects()

jSon_AlibabaSnapShot_AllDiskIDSMARTAttributesFirstDay = json.loads(doc_AlibabaSnapShot_AllDiskIDSMARTAttributesFirstDay.to_json())
df_AlibabaSnapShot_AllDiskIDSMARTAttributesFirstDay = pd.DataFrame.from_dict(jSon_AlibabaSnapShot_AllDiskIDSMARTAttributesFirstDay) 


# In[8]:


# Querying/Reading the FailuresAppsLocation database (mongodb) and turning It into a pandas dataframe - DB which merges both failures dataset

doc_MergeDs_FailuresAppsLocation = FailuresAppsLocation.objects()

jSon_MergeDs_FailuresAppsLocation = json.loads(doc_MergeDs_FailuresAppsLocation.to_json())
df_MergeDs_FailuresAppsLocation = pd.DataFrame.from_dict(jSon_MergeDs_FailuresAppsLocation) 

# Modifying the mongo db date type to some human-readable type
dicDateToString = json.dumps(list(df_MergeDs_FailuresAppsLocation['failure_time']))
dicStringToJson = json.loads(dicDateToString)
dicJsonToDf = pd.DataFrame.from_dict(dicStringToJson)
df_MergeDs_FailuresAppsLocation['failure_time'] = dicJsonToDf['$date']


# In[9]:


## Converting failure time columns to hours (for SSDs in common dataset)

# creating a temporary variable to be used to calculate the general mttf equation approach
df_genMTTF_MergeDs_FailuresAppsLocation = df_MergeDs_FailuresAppsLocation

## Value representing: 2018-01-01 00:00:00 (starting date from the experiment)
#1514764800000

#Subctracting by the initial time of the experiment and turning into hours
df_genMTTF_MergeDs_FailuresAppsLocation['failure_time'] = df_MergeDs_FailuresAppsLocation.failure_time.sub(1514764800000)
df_genMTTF_MergeDs_FailuresAppsLocation['failure_time'] = df_MergeDs_FailuresAppsLocation.failure_time.div(1000)
df_genMTTF_MergeDs_FailuresAppsLocation['failure_time'] = df_MergeDs_FailuresAppsLocation.failure_time.div(60)
df_genMTTF_MergeDs_FailuresAppsLocation['failure_time'] = df_MergeDs_FailuresAppsLocation.failure_time.div(60)
#df_failuresLocationDocumentsJsonMTTF.head(3)


# In[10]:


## Converting failure time columns to hours (for All failed SSDs dataset)

#Subctracting by the initial time of the experiment and turning into hours
df_AlibabaSnapShot_FailuresAppsLocation['failure_time'] = df_AlibabaSnapShot_FailuresAppsLocation.failure_time.sub(1514764800000)
df_AlibabaSnapShot_FailuresAppsLocation['failure_time'] = df_AlibabaSnapShot_FailuresAppsLocation.failure_time.div(1000)
df_AlibabaSnapShot_FailuresAppsLocation['failure_time'] = df_AlibabaSnapShot_FailuresAppsLocation.failure_time.div(60)
df_AlibabaSnapShot_FailuresAppsLocation['failure_time'] = df_AlibabaSnapShot_FailuresAppsLocation.failure_time.div(60)
#df_AlibabaSnapShot_FailuresAppsLocation.head()


# In[12]:


### To investigate the SSDs by application

exp_years = 2
exp_months = 12*exp_years
hours_per_year = 365*24
id_apps = df_AlibabaSnapShot_AllAppsSSDsLocation.app.unique()


#Creating a dataframe to generate some statistics taking into account the application which is running
df_idApps = pd.DataFrame(id_apps, columns=["app"])
#Adding empty colums
df_idAppsSSDs = pd.DataFrame(df_idApps, columns=["app", "N_AllSSDsApp", "N_failed_SDDs",  "AFR_SDDs", "mttf_SDDs", "AFR_SDDs_total", "mttf_SDDs_total", "N_failed_CommonSDDs",  "AFR_CommonSDDs", "mttf_CommonSDDs","AFR_CommonSDDs_total", "mttf_CommonSDDs_total"])


for i in id_apps:
    #Looping the AlibabaSnapshot_AllAppsSSDsLocation DB in order to count the total of ssds by application    
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_AllSSDsApp'] = df_AlibabaSnapShot_AllAppsSSDsLocation.loc[(df_AlibabaSnapShot_AllAppsSSDsLocation['app']) == i]['disk_id'].count()
    
    #Looping the FailuresAppsLocation (merged dss) in order to count the number of failed ssds by application    
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_failed_CommonSDDs'] = df_MergeDs_FailuresAppsLocation.loc[(df_MergeDs_FailuresAppsLocation['app']) == i]['disk_id'].count()
    #Looping the AllAppsSSDsFailures DB in order to count the number of failed ssds by application    
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_failed_SDDs'] = df_AlibabaSnapShot_FailuresAppsLocation.loc[(df_AlibabaSnapShot_FailuresAppsLocation['app']) == i]['disk_id'].count()
    
    # Calculating specific (failedSSDsperApp/TotalSSDsperAPP) app AFR and MTTF using failed ssds from the Alibaba Snapshot database (AllAppsSSDsFailures)
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_SDDs'] = ((df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_failed_SDDs'])*(12/exp_months))/(df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_AllSSDsApp'])
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'mttf_SDDs'] = (hours_per_year)/(df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_SDDs'])
    # Calculating specific (failedSSDsCommonperApp/TotalSSDsCommonperApp) app AFR and MTTF using failed ssds in common to the Snapshot and Overtime databases (FailuresAppsLocation)
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_CommonSDDs'] = ((df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_failed_CommonSDDs'])*(12/exp_months))/(df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_AllSSDsApp'])
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'mttf_CommonSDDs'] = (hours_per_year)/(df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_CommonSDDs'])
    
for i in id_apps:
    # Calculating total (failedSSDsperApp/TotalSSDsExperiment) AFR and MTTF using failed ssds from the Alibaba Snapshot database (AllAppsSSDsFailures)
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_SDDs_total'] = ((df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_failed_SDDs'])*(12/exp_months))/(df_idAppsSSDs.N_AllSSDsApp.sum())
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'mttf_SDDs_total'] = (hours_per_year)/(df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_SDDs_total'])
    # Calculating total (failedSSDsCommonperApp/TotalSSDsExperiment) AFR and MTTF using failed ssds in common to the Snapshot and Overtime databases (FailuresAppsLocation)
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_CommonSDDs_total'] = ((df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'N_failed_CommonSDDs'])*(12/exp_months))/(df_idAppsSSDs.N_AllSSDsApp.sum())
    df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'mttf_CommonSDDs_total'] = (hours_per_year)/(df_idAppsSSDs.loc[(df_idAppsSSDs['app']) == i, 'AFR_CommonSDDs_total'])

#Merging mttf per app to include the MTTF (this calculated using the general MTTF equation - not storage's specific) column into the failed ssds by application dataframe
df_idAppsSSDsMTTF = pd.merge(df_idAppsSSDs, df_app, how='left', on=['app'])
#df_idAppsSSDsMTTF.head(20)

df_idAppsSSDs.head(20)


# In[13]:


### To investigate the SDDs without app distinguishing

#Creating a data frame to calculate the AFR and MTTF without app distinguishing.
data = {'FailedSSDsSet':['FailedSSDs', 'FailedCommonSSDs']}
df_general_AFRMTTF = pd.DataFrame(data)
df_general_AFRMTTF = pd.DataFrame(df_general_AFRMTTF, columns=['FailedSSDsSet', 'AFR_General', 'MTTF_General'])

# Calculating general (failedSSDs/TotalSSDsExperiment) AFR and MTTF using failed ssds from the Alibaba Snapshot database (AllAppsSSDsFailures)
df_general_AFRMTTF.loc[(df_general_AFRMTTF['FailedSSDsSet']) == 'FailedSSDs', 'AFR_General'] = ((df_idAppsSSDs.N_failed_SDDs.sum())*(12/exp_months))/(df_idAppsSSDs.N_AllSSDsApp.sum())
df_general_AFRMTTF.loc[(df_general_AFRMTTF['FailedSSDsSet']) == 'FailedSSDs', 'MTTF_General'] = (hours_per_year)/(df_general_AFRMTTF.loc[(df_general_AFRMTTF['FailedSSDsSet']) == 'FailedSSDs', 'AFR_General'])
# Calculating general AFR and MTTF using failed ssds in common to the Snapshot and Overtime databases (FailuresAppsLocation)
df_general_AFRMTTF.loc[(df_general_AFRMTTF['FailedSSDsSet']) == 'FailedCommonSSDs', 'AFR_General'] = ((df_idAppsSSDs.N_failed_CommonSDDs.sum())*(12/exp_months))/(df_idAppsSSDs.N_AllSSDsApp.sum())
df_general_AFRMTTF.loc[(df_general_AFRMTTF['FailedSSDsSet']) == 'FailedCommonSSDs', 'MTTF_General'] = (hours_per_year)/(df_general_AFRMTTF.loc[(df_general_AFRMTTF['FailedSSDsSet']) == 'FailedCommonSSDs', 'AFR_General'])
df_general_AFRMTTF.head()
#df_idAppsSSDs.N_AllSSDsApp.sum()


# In[309]:


### To investigate the AFR and MTTF per SSD models for each application

exp_years = 2
exp_months = 12*exp_years
hours_per_year = 365*24
id_apps = df_AlibabaSnapShot_AllAppsSSDsLocation.app.unique()
id_models = df_AlibabaSnapShot_AllAppsSSDsLocation.model_y.unique()

# merge because alibaba's failure dataset doesn't contains the ssds models
df_MergeAlibabaAllandFailures =  pd.merge(df_AlibabaSnapShot_AllAppsSSDsLocation, df_AlibabaSnapShot_FailuresAppsLocation, how = 'inner', on = ['app', 'disk_id', 'node_id'])

df_idModelsSSDs = pd.DataFrame(columns=["app", "models", "N_AllSSDsApp", "N_failed_SDDs",  "AFR_SDDs", "mttf_SDDs"])

for j in id_models:
    for i in id_apps:
    
        #Looping the AlibabaSnapshot_AllAppsSSDsLocation DB in order to count the total of ssds/model by application    
        numberTotalSSDs = df_AlibabaSnapShot_AllAppsSSDsLocation.loc[(df_AlibabaSnapShot_AllAppsSSDsLocation.app.isin([i]) & df_AlibabaSnapShot_AllAppsSSDsLocation.model_y.isin([j])), "disk_id"].count()
        
        numberFailedSSDs = df_MergeAlibabaAllandFailures.loc[(df_MergeAlibabaAllandFailures.app.isin([i]) & df_MergeAlibabaAllandFailures.model_y.isin([j])), "disk_id"].count()

        if numberFailedSSDs != 0:
            AFR_SSDs = ((numberFailedSSDs)*(12/exp_months))/(numberTotalSSDs)
            MTTF_SSDs = (hours_per_year)/(AFR_SSDs)
        else: 
            AFR_SSDs = 0
            MTTF_SSDs = 0
        df_temp = pd.DataFrame(columns=["app", "models", "N_AllSSDsApp", "N_failed_SDDs",  "AFR_SDDs", "mttf_SDDs"])
        df_temp.loc[-1] = {'app' : i , 'models' : j, 'N_AllSSDsApp' : numberTotalSSDs, 'N_failed_SDDs': numberFailedSSDs, 'AFR_SDDs': AFR_SSDs, 'mttf_SDDs': MTTF_SSDs}
        df_idModelsSSDs = pd.concat([df_idModelsSSDs, df_temp], ignore_index=True)


df_idModelsSSDs.head(90)


# In[15]:


### To investigate the AFR and MTTF per SSD flash technology for each application

df_temp = df_idModelsSSDs.copy()
df_temp["tech"] = " "

df_temp.loc[(df_temp.models.isin(['A1']) | df_temp.models.isin(['A2']) | df_temp.models.isin(['A3']) | df_temp.models.isin(['A4']) | df_temp.models.isin(['A5']) | df_temp.models.isin(['A6']) | df_temp.models.isin(['B1']) | df_temp.models.isin(['B2']) | df_temp.models.isin(['B3'])), 'tech'] = 'MLC'
df_temp.loc[(df_temp.models.isin(['C1']) | df_temp.models.isin(['C2'])), 'tech'] = '3D-TLC'

df_temp1 = df_temp.groupby(['app', 'tech'], as_index=False)['N_failed_SDDs'].sum()
df_temp2 = df_temp.groupby(['app', 'tech'], as_index=False)['N_AllSSDsApp'].sum()
df_idTechSSDs = pd.merge(df_temp1, df_temp2, how='inner', on=['app', 'tech'])

df_idTechSSDs["AFR_SSDs"] = " "
df_idTechSSDs["mttf_SDDs"] = " "

for i in range(0, len(df_idTechSSDs)): 
    

    numberTotalSSDs = df_idTechSSDs.iloc[i].N_AllSSDsApp
    numberFailedSSDs = df_idTechSSDs.iloc[i].N_failed_SDDs
       
    if numberFailedSSDs != 0:
        AFR_SSDs = ((numberFailedSSDs)*(12/exp_months))/(numberTotalSSDs)
        MTTF_SSDs = (hours_per_year)/(AFR_SSDs)
    else: 
        AFR_SSDs = 0
        MTTF_SSDs = 0

    df_idTechSSDs.loc[i, 'AFR_SSDs'] = AFR_SSDs
    df_idTechSSDs.loc[i, 'mttf_SDDs'] = MTTF_SSDs

df_idTechSSDs.head(20)


# In[16]:


### To investigate the AFR and MTTF per SSD capacity for each application

df_temp = df_idModelsSSDs.copy()
df_temp["capacity"] = " "

df_temp.loc[(df_temp.models.isin(['A1']) | df_temp.models.isin(['A3']) | df_temp.models.isin(['A5']) | df_temp.models.isin(['B1'])), 'capacity'] = '480GB'
df_temp.loc[(df_temp.models.isin(['B2']) | df_temp.models.isin(['B3']) | df_temp.models.isin(['C1'])), 'capacity'] = '1920GB'
df_temp.loc[(df_temp.models.isin(['A2']) | df_temp.models.isin(['A6'])), 'capacity'] = '800GB'
df_temp.loc[(df_temp.models.isin(['A4'])), 'capacity'] = '240GB'
df_temp.loc[(df_temp.models.isin(['C2'])), 'capacity'] = '960GB'

df_temp1 = df_temp.groupby(['app', 'capacity'], as_index=False)['N_failed_SDDs'].sum()
df_temp2 = df_temp.groupby(['app', 'capacity'], as_index=False)['N_AllSSDsApp'].sum()
df_idCapacitySSDs = pd.merge(df_temp1, df_temp2, how='inner', on=['app', 'capacity'])

df_idCapacitySSDs["AFR_SSDs"] = " "
df_idCapacitySSDs["mttf_SDDs"] = " "

for i in range(0, len(df_idCapacitySSDs)): 
    

    numberTotalSSDs = df_idCapacitySSDs.iloc[i].N_AllSSDsApp
    numberFailedSSDs = df_idCapacitySSDs.iloc[i].N_failed_SDDs
       
    if numberFailedSSDs != 0:
        AFR_SSDs = ((numberFailedSSDs)*(12/exp_months))/(numberTotalSSDs)
        MTTF_SSDs = (hours_per_year)/(AFR_SSDs)
    else: 
        AFR_SSDs = 0
        MTTF_SSDs = 0

    df_idCapacitySSDs.loc[i, 'AFR_SSDs'] = AFR_SSDs
    df_idCapacitySSDs.loc[i, 'mttf_SDDs'] = MTTF_SSDs

df_idCapacitySSDs.head(20)


# In[17]:


### To investigate the AFR and MTTF per SSD lithography for each application

df_temp = df_idModelsSSDs.copy()
df_temp["lithography"] = " "

df_temp.loc[(df_temp.models.isin(['A1']) | df_temp.models.isin(['A2']) | df_temp.models.isin(['A3']) | df_temp.models.isin(['A6'])), 'lithography'] = '20nm'
df_temp.loc[(df_temp.models.isin(['A4']) | df_temp.models.isin(['A5'])), 'lithography'] = '16nm'
df_temp.loc[(df_temp.models.isin(['C1']) | df_temp.models.isin(['C2'])), 'lithography'] = 'V1'
df_temp.loc[(df_temp.models.isin(['B1'])), 'lithography'] = '21nm'
df_temp.loc[(df_temp.models.isin(['B2'])), 'lithography'] = '19nm'
df_temp.loc[(df_temp.models.isin(['B3'])), 'lithography'] = '24nm'

df_temp1 = df_temp.groupby(['app', 'lithography'], as_index=False)['N_failed_SDDs'].sum()
df_temp2 = df_temp.groupby(['app', 'lithography'], as_index=False)['N_AllSSDsApp'].sum()
df_idLitSSDs = pd.merge(df_temp1, df_temp2, how='inner', on=['app', 'lithography'])

df_idLitSSDs["AFR_SSDs"] = " "
df_idLitSSDs["mttf_SDDs"] = " "

for i in range(0, len(df_idLitSSDs)): 
    

    numberTotalSSDs = df_idLitSSDs.iloc[i].N_AllSSDsApp
    numberFailedSSDs = df_idLitSSDs.iloc[i].N_failed_SDDs
       
    if numberFailedSSDs != 0:
        AFR_SSDs = ((numberFailedSSDs)*(12/exp_months))/(numberTotalSSDs)
        MTTF_SSDs = (hours_per_year)/(AFR_SSDs)
    else: 
        AFR_SSDs = 0
        MTTF_SSDs = 0

    df_idLitSSDs.loc[i, 'AFR_SSDs'] = AFR_SSDs
    df_idLitSSDs.loc[i, 'mttf_SDDs'] = MTTF_SSDs

df_idLitSSDs.head(20)


# In[401]:


df_idLitSSDs.info()


# In[331]:


### To investigate the number of blocks written/read per app - For all SSDs

# merging the last day experiment smart attributes to the dataframe containing various statistics regarding apps
df_temp = pd.merge(df_AlibabaSnapShot_AllDiskIDSMARTAttributes, df_AlibabaSnapShot_AllAppsSSDsLocation, how='inner', on=['disk_id', 'model_y'])

# total number of write/read blocks and median wearout indicator per application
df_temp_blocks_written_app = df_temp.groupby(['app'], as_index=False)['n_b_written'].sum()
df_temp_blocks_read_app = df_temp.groupby(['app'], as_index=False)['n_b_read'].sum()
df_temp_media_wearout = df_temp.groupby(['app'], as_index=False)['media_wearout_i'].mean()
df_temp_media_wearlevelingcount = df_temp.groupby(['app'], as_index=False)['w_l_count'].mean()
df_temp_media_wearrangedelta = df_temp.groupby(['app'], as_index=False)['w_r_d'].mean()

df_temp_blocks_writtenread_app = pd.merge(df_temp_blocks_written_app, df_temp_blocks_read_app, how='inner', on=['app'])
df_temp_blocks_writtenreadwear_app = pd.merge(df_temp_blocks_writtenread_app, df_temp_media_wearout, how='inner', on=['app'])
df_temp_blocks_writtenreadwear_app = pd.merge(df_temp_blocks_writtenreadwear_app, df_temp_media_wearlevelingcount, how='inner', on=['app'])
df_temp_blocks_writtenreadwear_app = pd.merge(df_temp_blocks_writtenreadwear_app, df_temp_media_wearrangedelta, how='inner', on=['app'])
df_temp_blocks_writtenreadwear_app = pd.merge(df_temp_blocks_writtenreadwear_app, df_idAppsSSDs, how='inner', on=['app'])

df_temp_blocks_writtenreadwear_app["%write"] = ((df_temp_blocks_writtenreadwear_app["n_b_written"])/(df_temp_blocks_writtenreadwear_app["n_b_read"]+df_temp_blocks_writtenreadwear_app["n_b_written"]))*100
df_temp_blocks_writtenreadwear_app["%read"] = ((df_temp_blocks_writtenreadwear_app["n_b_read"])/(df_temp_blocks_writtenreadwear_app["n_b_read"]+df_temp_blocks_writtenreadwear_app["n_b_written"]))*100

df_temp_blocks_writtenreadwear_app = df_temp_blocks_writtenreadwear_app.fillna(0)    

df_temp_blocks_writtenreadwear_app.head(10)


# In[438]:


### To investigate the number of blocks written/read per app for distintic ssd models - For all SSDs

# merging the last day experiment smart attributes to the dataframe containing various statistics regarding apps
df_temp = pd.merge(df_AlibabaSnapShot_AllDiskIDSMARTAttributes, df_AlibabaSnapShot_AllAppsSSDsLocation, how='inner', on=['disk_id', 'model_y'])

# total number of write/read blocks and median wear out indicators per application
df_temp_blocks_written_app = df_temp.groupby(['app', 'model_y'], as_index=False)['n_b_written'].sum()
df_temp_blocks_read_app = df_temp.groupby(['app', 'model_y'], as_index=False)['n_b_read'].sum()
df_temp_reallocated_sectors = df_temp.groupby(['app', 'model_y'], as_index=False)['r_sectors'].mean()
df_temp_power_on = df_temp.groupby(['app', 'model_y'], as_index=False)['p_on'].mean()
df_temp_media_wearout = df_temp.groupby(['app', 'model_y'], as_index=False)['media_wearout_i'].mean()
df_temp_media_wearlevelingcount = df_temp.groupby(['app', 'model_y'], as_index=False)['w_l_count'].mean()
df_temp_media_wearrangedelta = df_temp.groupby(['app', 'model_y'], as_index=False)['w_r_d'].mean()

df_temp_blocks_writtenreadwear_appModel = pd.merge(df_temp_blocks_written_app, df_temp_blocks_read_app, how='inner', on=['app', 'model_y'])
df_temp_blocks_writtenreadwear_appModel = pd.merge(df_temp_blocks_writtenreadwear_appModel, df_temp_reallocated_sectors, how='inner', on=['app', 'model_y'])
df_temp_blocks_writtenreadwear_appModel = pd.merge(df_temp_blocks_writtenreadwear_appModel, df_temp_power_on, how='inner', on=['app', 'model_y'])
df_temp_blocks_writtenreadwear_appModel = pd.merge(df_temp_blocks_writtenreadwear_appModel, df_temp_media_wearout, how='inner', on=['app', 'model_y'])
df_temp_blocks_writtenreadwear_appModel = pd.merge(df_temp_blocks_writtenreadwear_appModel, df_temp_media_wearlevelingcount, how='inner', on=['app','model_y'])
df_temp_blocks_writtenreadwear_appModel = pd.merge(df_temp_blocks_writtenreadwear_appModel, df_temp_media_wearrangedelta, how='inner', on=['app','model_y'])

df_temp_blocks_writtenreadwear_appModel["%write"] = ((df_temp_blocks_writtenreadwear_appModel["n_b_written"])/(df_temp_blocks_writtenreadwear_appModel["n_b_read"]+df_temp_blocks_writtenreadwear_appModel["n_b_written"]))*100
df_temp_blocks_writtenreadwear_appModel["%read"] = ((df_temp_blocks_writtenreadwear_appModel["n_b_read"])/(df_temp_blocks_writtenreadwear_appModel["n_b_read"]+df_temp_blocks_writtenreadwear_appModel["n_b_written"]))*100


for i in id_apps:
    totalWrittenBlocks = (df_temp_blocks_writtenreadwear_appModel.loc[df_temp_blocks_writtenreadwear_appModel['app'] == i]).n_b_written.sum()
    df_temp_blocks_writtenreadwear_appModel.loc[df_temp_blocks_writtenreadwear_appModel['app'] == i, "%write_appModel"] = ((df_temp_blocks_writtenreadwear_appModel["n_b_written"])/(totalWrittenBlocks))*100

df_temp_blocks_writtenreadwear_appModel = df_temp_blocks_writtenreadwear_appModel.fillna(0)    

# checking the resulting dataframe
df_temp_blocks_writtenreadwear_appModel.head(20)


# In[439]:


### Replacing NaN values with zero

df_temp_blocks_writtenreadwear_appModel = df_temp_blocks_writtenreadwear_appModel.fillna(0)


# In[445]:


### Creating a single dataframe to export it and to perform statics analsys as DOE

df_Temp1 = df_temp_blocks_writtenreadwear_appModel.copy()
df_Temp1 = pd.DataFrame(df_Temp1)
df_Temp1.rename(columns = {'model_y':'models'}, inplace=True)
df_Temp2 = df_idModelsSSDs.copy()
df_Temp2 = pd.DataFrame(df_Temp2)

df_DataSetDoe = pd.merge(df_Temp1, df_Temp2, how='outer', on=['app', 'models'])

df_temp3 = df_DataSetDoe.copy()
df_temp3["lithography"] = " "

df_temp3.loc[(df_temp3.models.isin(['A1']) | df_temp3.models.isin(['A2']) | df_temp3.models.isin(['A3']) | df_temp3.models.isin(['A6'])), 'lithography'] = '20nm'
df_temp3.loc[(df_temp3.models.isin(['A4']) | df_temp3.models.isin(['A5'])), 'lithography'] = '16nm'
df_temp3.loc[(df_temp3.models.isin(['C1']) | df_temp3.models.isin(['C2'])), 'lithography'] = 'V1'
df_temp3.loc[(df_temp3.models.isin(['B1'])), 'lithography'] = '21nm'
df_temp3.loc[(df_temp3.models.isin(['B2'])), 'lithography'] = '19nm'
df_temp3.loc[(df_temp3.models.isin(['B3'])), 'lithography'] = '24nm'

df_temp4 = df_temp3.copy()
df_temp4["capacity"] = " "

df_temp4.loc[(df_temp4.models.isin(['A1']) | df_temp4.models.isin(['A3']) | df_temp4.models.isin(['A5']) | df_temp4.models.isin(['B1'])), 'capacity'] = '480GB'
df_temp4.loc[(df_temp4.models.isin(['B2']) | df_temp4.models.isin(['B3']) | df_temp4.models.isin(['C1'])), 'capacity'] = '1920GB'
df_temp4.loc[(df_temp4.models.isin(['A2']) | df_temp4.models.isin(['A6'])), 'capacity'] = '800GB'
df_temp4.loc[(df_temp4.models.isin(['A4'])), 'capacity'] = '240GB'
df_temp4.loc[(df_temp4.models.isin(['C2'])), 'capacity'] = '960GB'

df_temp5 = df_temp4.copy()
df_temp5["tech"] = " "

df_temp5.loc[(df_temp5.models.isin(['A1']) | df_temp5.models.isin(['A2']) | df_temp5.models.isin(['A3']) | df_temp5.models.isin(['A4']) | df_temp5.models.isin(['A5']) | df_temp5.models.isin(['A6']) | df_temp5.models.isin(['B1']) | df_temp5.models.isin(['B2']) | df_temp5.models.isin(['B3'])), 'tech'] = 'MLC'
df_temp5.loc[(df_temp5.models.isin(['C1']) | df_temp5.models.isin(['C2'])), 'tech'] = '3D-TLC'

df_DataSetDoe = df_temp5.copy()
df_DataSetDoe = pd.DataFrame(df_DataSetDoe)


# In[455]:


## Checking the dataframe created for storing tabulated information about all SSDs, applications and attributes

df_DataSetDoe.head(50)


# In[449]:


## Saving the information from both datasets tabulated in a csv.

df_DataSetDoe = df_DataSetDoe.fillna(0)
df_DataSetDoe["AFR_SDDs"] = df_DataSetDoe["AFR_SDDs"]*100
df_DataSetDoe.to_csv('DataSetDoe.csv')


# In[2]:


## Loading tabulated data from a csv to conduct experiments

df_loadDoE = pd.read_csv('DataSetDoe.csv')


# In[5]:


## Testing (checking that the data has been loaded correctly)

df_loadDoE.head()


# In[5]:


# Testing (making a query filtering some factors)

x = df_loadDoE.loc[(df_loadDoE.app.isin(["none"])) & (df_loadDoE.models.isin(["B2"]))]
print(x)


# In[3]:


## ### Plotting the AFRs per app - considering #ssds per app 

df_plotAFRMTTFSPF = df_loadDoE.groupby(['app'], as_index=False)['AFR_SDDs'].mean()

# Customize plotting parameters
font_size = 36
title_size = 18
label_size = 36

plt.figure(figsize=(13, 6))
graph = sns.barplot(x = "app", y = 'AFR_SDDs', palette = 'ch:.25', data = df_plotAFRMTTFSPF, order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
graph.axhline(df_plotAFRMTTFSPF.AFR_SDDs.mean())
#graph.set_ylabel("Gender",size = 67,color="g",alpha=0.5)
graph.set_ylabel("AFR (%)", fontsize=label_size)
plt.xlabel('application', fontsize=label_size)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
plt.tight_layout()
plt.grid(False)
plt.savefig('img/afrappssd.pdf', bbox_inches='tight')
sns.despine()
plt.show()


# In[34]:


## Investigating the number of blocks written and AFR per SSD model

# Data Preparation
# Sum of n_b_written for each model
sum_n_b_written = df.groupby('models')['n_b_written'].sum().reset_index()

# Mean of AFR_SDDs for each model
mean_AFR_SDDs = df.groupby('models')['AFR_SDDs'].mean().reset_index()

# Customize plotting parameters
font_size = 36
title_size = 18
label_size = 36

# Plotting
# Plot 1: Sum of n_b_written for each model
plt.figure(figsize=(12, 7))
sns.barplot(x='models', y='n_b_written', data=sum_n_b_written, palette='ch:.25')
#plt.xticks(rotation=45)
#plt.title('Sum of n_b_written for Each Model')
plt.xlabel('SSD models', fontsize=label_size)
plt.ylabel('#Blocks written', fontsize=label_size)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
#plt.legend(fontsize=23)
plt.tight_layout()
plt.grid(False)
# Adjust the font size of the x-axis order of magnitude
ax = plt.gca()
ax.yaxis.get_offset_text().set_fontsize(label_size-5)
plt.savefig('img/modelwrittenblocks.pdf', bbox_inches='tight')
plt.show()

# Plot 2: Mean of AFR_SDDs for each model
plt.figure(figsize=(12, 7))
sns.barplot(x='models', y='AFR_SDDs', data=mean_AFR_SDDs, palette='ch:.25')
#plt.xticks(rotation=45)
#plt.title('Mean of AFR_SDDs for Each Model')
plt.xlabel('SSD models', fontsize=label_size)
plt.ylabel('AFR (%)', fontsize=label_size)
plt.xticks(fontsize=font_size-2)
plt.yticks(fontsize=font_size-2)
#plt.legend(fontsize=23)
plt.tight_layout()
plt.grid(False)
plt.savefig('img/modelAFR.pdf', bbox_inches='tight')
plt.show()


# In[151]:


## Investigating the number of failed SSDs per application

df_plotNSDDsFailedTotal1 = df_idAppsSSDs.loc[:,['app','N_AllSSDsApp']]
df_plotNSDDsFailedTotal2 = df_idAppsSSDs.loc[:,['app','N_failed_SDDs']]

df_plotNSDDsFailedTotal1 = df_plotNSDDsFailedTotal1.assign(group='Total')
df_plotNSDDsFailedTotal2 = df_plotNSDDsFailedTotal2.assign(group='Failed')
df_plotNSDDsFailedTotal2.rename(columns = {'N_failed_SDDs':'N_AllSSDsApp'}, inplace=True)

df_plotNSDDsFailedTotal1 = pd.concat([df_plotNSDDsFailedTotal1,df_plotNSDDsFailedTotal2])

graph = sns.barplot(x = "app", y = 'N_AllSSDsApp', palette = 'ch:.25', hue = 'group', data = df_plotNSDDsFailedTotal1, order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge')
graph.set_ylabel("#SSDs")
graph.figure.set_figwidth(11.27) # increasing the figure width
#graph.fig.set_figheight(11.7)
sns.despine()
plt.show()


# In[327]:


## Investigating the AFR per application for each SSD model

df_temp = df_idModelsSSDs.copy()
df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

graph = sns.barplot(x = "app", y = 'AFR_SDDs', palette = 'Spectral', hue = 'models', data = df_temp, hue_order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'], order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

graph.axhline(df_general_AFRMTTF.loc[df_general_AFRMTTF['FailedSSDsSet'] == 'FailedSSDs', 'AFR_General'].values[0]*100)
graph.set_ylabel("AFR (%)")
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figwidth(20.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()



# In[452]:


## Investigating the AFR per SSD models for each application

df_temp = df_idModelsSSDs.copy()
df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

graph = sns.barplot(x = "models", y = 'AFR_SDDs', palette = 'Spectral', hue = 'app', data = df_temp, order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'], hue_order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

graph.axhline(df_general_AFRMTTF.loc[df_general_AFRMTTF['FailedSSDsSet'] == 'FailedSSDs', 'AFR_General'].values[0]*100)
graph.set_ylabel("AFR (%)")
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figwidth(20.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[299]:


## Investigating the AFR per SSD models for each application using stacking approach (to get a different perspective)

df_temp = df_idModelsSSDs.copy()
df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

order1 = ['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none']
df_temp.app = df_temp.app.astype("category")
df_temp.app.cat.set_categories(order1, inplace=True)
df_temp.sort_values(by='app')

ax = sns.histplot(
    df_temp,
    x='app',
    # Use the value variable here to turn histogram counts into weighted
    # values.
    weights='AFR_SDDs',
    hue='models',
    #multiple='dodge',   bars
    #multiple='layer',
    multiple='stack',
    #multiple='fill',
    palette='Spectral',
    #palette='icefire',
    #palette='husl',
    # Add white borders to the bars.
    #edgecolor='white',
    linewidth=1.2,
    edgecolor=".2",
    # Shrink the bars a bit so they don't touch.
    shrink=0.85,
    hue_order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'],
    
)
#ax.axhline(df_general_AFRMTTF.loc[df_general_AFRMTTF['FailedSSDsSet'] == 'FailedSSDs', 'AFR_General'].values[0]*100)
ax.figure.set_figwidth(14.87) # increasing the figure width
ax.figure.set_figheight(7.7)
#ax.set_title('Tips by Day and Gender')
# Remove 'Count' ylabel.
ax.set_ylabel("AFR (%)")


# In[350]:


## Investigating the number of SSD models for each application

df_temp = df_idModelsSSDs.copy()
#df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

graph = sns.barplot(x = "app", y = 'N_AllSSDsApp', palette = 'Spectral', hue = 'models', data = df_temp, hue_order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'], order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

graph.axhline(df_idModelsSSDs.N_AllSSDsApp.mean())
graph.set_ylabel("#SSDs")
graph.figure.set_figwidth(22.87) # increasing the figure width
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[361]:


df_idModelsSSDs.head(20)


# In[355]:


### Investigating the number of devices per SSD models for each application

df_temp = df_idModelsSSDs.copy()
#df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

graph = sns.barplot(x = "models", y = 'N_AllSSDsApp', palette = 'Spectral', hue = 'app', data = df_temp, order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'], hue_order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

#graph.axhline(df_idModelsSSDs.N_AllSSDsApp.mean())
graph.set_ylabel("#SSDs")
graph.figure.set_figwidth(22.87) # increasing the figure width
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[349]:


## Investigating the number of failed SSD models for each application

df_temp = df_idModelsSSDs.copy()
#df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

graph = sns.barplot(x = "app", y = 'N_failed_SDDs', palette = 'Spectral', hue = 'models', data = df_temp, hue_order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'], order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

graph.axhline(df_idModelsSSDs.N_failed_SDDs.mean())
graph.set_ylabel("#Failed SSDs")
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figwidth(22.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[356]:


## Investigating the number of failed devices per SSD models for each application

df_temp = df_idModelsSSDs.copy()
#df_temp["AFR_SDDs"] = df_temp["AFR_SDDs"]*100

graph = sns.barplot(x = "models", y = 'N_failed_SDDs', palette = 'Spectral', hue = 'app', data = df_temp, order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'], hue_order=['DAE', 'DB', 'NAS', 'RM', 'SS', 'WPS', 'WS', 'WSM', 'none'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

#graph.axhline(df_idModelsSSDs.N_failed_SDDs.mean())
graph.set_ylabel("#Failed SSDs")
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figwidth(22.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[239]:


### Investigating the AFR per SSD flash technology for each application

df_temp3Plot = df_idTechSSDs.copy()
df_temp3Plot["AFR_SSDs"] = df_temp3Plot["AFR_SSDs"]*100
graph = sns.barplot(x = "app", y = 'AFR_SSDs', palette = 'Set2', hue = 'tech', data = df_temp3Plot, hue_order = ['MLC','3D-TLC'])

for container in graph.containers:
    graph.bar_label(container, label_type='edge')


graph.set_ylabel("AFR (%)")
graph.axhline(df_general_AFRMTTF.loc[df_general_AFRMTTF['FailedSSDsSet'] == 'FailedSSDs', 'AFR_General'].values[0]*100)
graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[240]:


## Investigating the AFR per SSD flash capacity for each application

df_temp3Plot = df_idCapacitySSDs.copy()
df_temp3Plot["AFR_SSDs"] = df_temp3Plot["AFR_SSDs"]*100
graph = sns.barplot(x = "app", y = 'AFR_SSDs', palette = 'Spectral', hue = 'capacity', data = df_temp3Plot, hue_order = ['240GB','480GB','800GB','960GB','1920GB'])

#for container in graph.containers:
#    graph.bar_label(container, label_type='edge')


graph.set_ylabel("AFR (%)")
graph.axhline(df_general_AFRMTTF.loc[df_general_AFRMTTF['FailedSSDsSet'] == 'FailedSSDs', 'AFR_General'].values[0]*100)
graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[244]:


## Investigating the AFR per SSD flash lithography for each application

df_temp3Plot = df_idLitSSDs.copy()
df_temp3Plot["AFR_SSDs"] = df_temp3Plot["AFR_SSDs"]*100
graph = sns.barplot(x = "app", y = 'AFR_SSDs', palette = 'Spectral', hue = 'lithography', data = df_temp3Plot, hue_order = ['16nm','19nm','20nm','21nm','24nm','V1'])

#for container in graph.containers:
#    graph.bar_label(container, label_type='edge')


graph.set_ylabel("AFR (%)")
graph.axhline(df_general_AFRMTTF.loc[df_general_AFRMTTF['FailedSSDsSet'] == 'FailedSSDs', 'AFR_General'].values[0]*100)
graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[102]:


## Investigating SMART attributes per application

df_temp = df_temp_blocks_writtenreadwear_app.copy()

#graph = sns.barplot(x = "app", y = 'n_b_written', palette = 'Spectral', data = df_temp)
#graph = sns.barplot(x = "app", y = 'n_b_read', palette = 'Spectral', data = df_temp)

graph, axs = plt.subplots(2, 3, figsize=(17, 7))

graph1 = sns.barplot(data=df_temp, x="app", y='n_b_written', palette='ch:.25', ax=axs[0,0])
graph2 = sns.barplot(data=df_temp, x="app", y='n_b_read', palette='ch:.25', ax=axs[0,1])
graph3 = sns.barplot(data=df_temp, x="app", y='%write', palette='ch:.25', ax=axs[0,2])
graph4 = sns.barplot(data=df_temp, x="app", y='media_wearout_i', palette='ch:.25', ax=axs[1,0])
graph5 = sns.barplot(data=df_temp, x="app", y='w_l_count', palette='ch:.25', ax=axs[1,1])
graph6 = sns.barplot(data=df_temp, x="app", y='w_r_d', palette='ch:.25', ax=axs[1,2])
#sns.histplot(data=df_temp, x="petal_length", kde=True, color="gold", ax=axs[1, 0])
#sns.histplot(data=df_temp, x="petal_width", kde=True, color="teal", ax=axs[1, 1])

for container in graph3.containers:
    graph3.bar_label(container, label_type='edge')

graph1.axhline((df_temp['n_b_written'].mean()))
graph2.axhline((df_temp['n_b_read'].mean()))
graph3.axhline((df_temp['%write'].mean()))
graph4.axhline((df_temp['media_wearout_i'].mean()))
graph5.axhline((df_temp['w_l_count'].mean()))
graph6.axhline((df_temp['w_r_d'].mean()))
graph1.set_ylabel("#written_blocks")
graph2.set_ylabel("#read_blocks")
graph3.set_ylabel("Write (%)")
graph4.set_ylabel("Avg. Media Wearout Indicator")
graph5.set_ylabel("Avg. Wear Leveling Count")
graph6.set_ylabel("Avg. Wear Range Delta")
#graph.figure.set_figwidth(14.87) # increasing the figure width
#graph.figure.set_figheight(5.7)
#[graph1.axvline(x+.5,color='silver') for x in graph1.get_xticks()]
sns.despine()
plt.show()



# In[342]:


## Investigating the percentage of requests that are "write" by application

df_temp = df_temp_blocks_writtenreadwear_app.copy()
#df_temp['n_b_written'] = (df_temp['n_b_written'])/(math.exp(1006))

graph = sns.barplot(x = "app", y = '%write', palette = 'Spectral', data = df_temp)
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='horizontal', padding=5)

graph.axhline((df_temp_blocks_writtenreadwear_app['%write'].mean()))
graph.set_ylabel("Write (%)")
graph.figure.set_figwidth(14.87) # increasing the figure width
#graph.figure.set_figwidth(22.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#plt.legend(['models'])
#sns.despine()
plt.show()


# In[333]:


## Investigating the number of write blocks per application for each SSD model

df_temp = df_temp_blocks_writtenreadwear_appModel.copy()
#df_temp['n_b_written'] = (df_temp['n_b_written'])/(math.exp(1006))

graph = sns.barplot(x = "app", y = 'n_b_written', palette = 'Spectral', hue = 'model_y', data = df_temp, hue_order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

graph.axhline((df_temp_blocks_writtenreadwear_app['n_b_written'].mean()))
graph.set_ylabel("#written_blocks")
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figwidth(22.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#plt.legend(['models'])
#sns.despine()
plt.show()



# In[357]:


## Investigating the number of write blocks per SSD model for each application

df_temp = df_temp_blocks_writtenreadwear_appModel.copy()
#df_temp['n_b_written'] = (df_temp['n_b_written'])/(math.exp(1006))

graph = sns.barplot(x = "model_y", y = 'n_b_written', palette = 'Spectral', hue = 'app', data = df_temp, order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

#graph.axhline((df_temp_blocks_writtenreadwear_app['n_b_written'].mean()))
graph.set_ylabel("#written_blocks")
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figwidth(22.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#plt.legend(['models'])
#sns.despine()
plt.show()


# In[359]:


## Investigating the percentage of requests that are "write" per SSD model for each application

df_temp = df_temp_blocks_writtenreadwear_appModel.copy()

graph = sns.barplot(x = "model_y", y = '%write_appModel', palette = 'Spectral', hue = 'app', data = df_temp, order = ['A1','A2','A3','A4','A5','A6','B1','B2','B3','C1','C2'])
for container in graph.containers:
    graph.bar_label(container, label_type='edge', rotation='vertical', padding=5)

#graph.axhline((df_temp_blocks_writtenreadwear_app['%write'].mean()))
graph.set_ylabel("Write(%)")
graph.figure.set_figwidth(22.87) # increasing the figure width
#graph.figure.set_figwidth(14.87) # increasing the figure width
graph.figure.set_figheight(5.7)
[graph.axvline(x+.5,color='silver') for x in graph.get_xticks()]
#sns.despine()
plt.show()


# In[ ]:


## Saving sorted failure time per app into csv to be used for probability distribution fitting (All failed SDDs)

for i in id_apps:
    
    df_alibabasnapshot_toFitting = df_AlibabaSnapShot_FailuresAppsLocation.loc[df_AlibabaSnapShot_FailuresAppsLocation["app"] == i, "failure_time"]
    df_alibabasnapshot_toFitting.sort_values
    df_alibabasnapshot_toFitting.to_csv('Snapshot_FailureTimesApp' + i + '.csv', header=None, index=None)

df_alibabasnapshot_toFitting = df_AlibabaSnapShot_FailuresAppsLocation['failure_time']
df_alibabasnapshot_toFitting.to_csv('Snapshot_FailureTimesApp' + 'All' + '.csv', header=None, index=None)


# In[ ]:


## Saving sorted failure time per app into csv to be used for probability distribution fitting (SSDs in common)

for i in id_apps:
    
    df_merge_toFitting_times = df_genMTTF_MergeDs_FailuresAppsLocation.loc[df_genMTTF_MergeDs_FailuresAppsLocation["app"] == i, "failure_time"]
    df_merge_toFitting_times.sort_values
    df_merge_toFitting_times.to_csv('MergeSSDs_FailureTimesApp' + i + '.csv', header=None, index=None)

