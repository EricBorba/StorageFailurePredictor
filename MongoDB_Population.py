#!/usr/bin/env python
# coding: utf-8

# In[1]:


## reset specific variables (replace regular_expression by the variables of interest)
#%reset_selective <regular_expression>

# reset all variables
get_ipython().run_line_magic('reset', '-f')


# In[2]:


## Importing libraries

from datetime import datetime, date, timedelta
from pathlib import Path
from pylab import savefig
import pandas as pd
import csv
from pymongo import MongoClient
from mongoengine import *


# In[2]:


## Creating/Connecting Mongo DB instances

# Provide the mongodb atlas url to connect python to mongodb using pymongo
#CONNECTION_STRING = "mongodb+srv://<jgu>:<123>@<cluster-jgu>.mongodb.net/SMARTAttributesFilter"

connect(db='SMARTAttributesFilter', alias='SMARTAttributesFilter_alias')

connect(db='SMARTAttributesFilterOverWear', alias='SMARTAttributesFilterOverWear_alias')

connect(db='SMARTAttributesFilterFull', alias='SMARTAttributesFilterFull_alias')

connect(db='OverTimeSSDsFailures', alias='OverTimeSSDsFailures_alias')

connect(db='AllAppsSSDsFailures', alias='AllAppsSSDsFailures_alias')

connect(db='AllAppsSSDsLocation', alias='AllAppsSSDsLocation_alias')

connect(db='AllDiskIDSMARTAttributes', alias='AllDiskIDSMARTAttributes_alias')

connect(db='AllDiskIDSMARTAttributesFirstDay', alias='AllDiskIDSMARTAttributesFirstDay_alias')

connect(db='FailuresAppsLocation', alias='FailuresAppsLocation_alias')

connect(db='SMARTAtt_FailuresAppsLocation', alias='SMARTAtt_FailuresAppsLocation_alias')

connect(db='SMARTAttFullBackBlaze', alias='SMARTAttFullBackBlaze_alias')


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


# SMART attributes and disk information from the 2 years daily sampling dataset (500k disks)
class SMARTAttFull(Document):
     disk_id = FloatField(required=False, default='0')
     timestamp = DateTimeField(required=False, default='0')
     model_x = StringField(required=False, default='0')
     r_sectors = FloatField(required=False, default='0')
     power_hours = FloatField(required=False, default='0')
     u_errors = FloatField(required=False, default='0')
     p_failedA = FloatField(required=False, default='0')
     p_failedB = FloatField(required=False, default='0')
     e_failedA = FloatField(required=False, default='0')
     e_failedB = FloatField(required=False, default='0')
     n_b_written = FloatField(required=False, default='0')
     n_b_read = FloatField(required=False, default='0')
     w_l_count = FloatField(required=False, default='0')
     w_r_d = FloatField(required=False, default='0')
     media_wearout_i = FloatField(required=False, default='0')
     meta = {'db_alias': 'SMARTAttributesFilterFull_alias'}


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

class SMARTAttFullBackBlaze(Document):
     timestamp = DateTimeField(required=False, default='0')
     disk_id = StringField(required=False, default='0')
     model = StringField(required=False, default='0')
     capacity_bytes = FloatField(required=False, default='0')
     failure = FloatField(required=False, default='0')
     r_sectors = FloatField(required=False, default='0')
     power_hours = FloatField(required=False, default='0')
     u_errors = FloatField(required=False, default='0')
     read_error_rate = FloatField(required=False, default='0')
     n_b_written = FloatField(required=False, default='0')
     n_b_read = FloatField(required=False, default='0')
     command_timeout = FloatField(required=False, default='0')
     current_pending_sector_count = FloatField(required=False, default='0')
     uncorrectable_sector_count = FloatField(required=False, default='0')
     meta = {'db_alias': 'SMARTAttFullBackBlaze_alias'}


# In[ ]:


## Deleting DB content (for the case when the goal is to test the code for filling from zero)

# Creating the object related to the whole collection
failuresAppsLocationTeste = FailuresAppsLocation.objects() 

# Deleting all collection
failuresAppsLocationTeste.delete() 


# In[12]:


# Closing the connection to the DB

disconnect(alias='SMARTAttributesFilter_alias')

disconnect(alias='SMARTAttributesFilterOverWear_alias')

disconnect(alias='OverTimeSSDsFailures_alias')

disconnect(alias='AllAppsSSDsFailures_alias')

disconnect(alias='AllAppsSSDsLocation_alias')

disconnect(alias='AllDiskIDSMARTAttributes_alias')

disconnect(alias='FailuresAppsLocation_alias')

disconnect(alias='SMARTAtt_FailuresAppsLocation_alias')


# In[42]:


## Loading dataset - AlibabaOverTime (Failurelogs)

df_AlibabaOver_Failurelogs = pd.read_csv('/media/erb/hdd1/DataSet/alibabaOvertime/ssd_failure_label/ssd_failure_label.csv')


# In[43]:


## Loading dataset - Alibaba Snapshot (TimeStamps of failed SSDs, SMART attributes in 39 columns, SSDs location, applications, SSD models and Disk ID)

df_AlibabaSnapShot_FailuresAppsLocation = pd.read_csv('/media/erb/hdd1/DataSet/alibabaSnapShot/ssd_failure_tag/ssd_failure_tag.csv')


# In[44]:


## Loading dataset - Alibaba Snapshot (Location of all SSDs, applications, SSD models, rack and Disk ID)

df_AlibabaSnapShot_AllAppsSSDsLocation = pd.read_csv('/media/erb/hdd1/DataSet/alibabaSnapShot/location_info_of_ssd/location_info_of_ssd.csv')


# In[49]:


## Loading dataset - Alibaba Snapshot (SMART attributes)

df_AlibabaSnapShot_SMARTAttributes = pd.read_csv('/media/erb/hdd1/DataSet/alibabaSnapShot/smart_log_20191231/20191231.csv')


# In[50]:


## Loading dataset - Alibaba Snapshot (SMART attributes)

df_AlibabaSnapShot_SMARTAttributesFirstDay = pd.read_csv('/media/erb/hdd1/DataSet/alibabaOvertime/smartlogs/20180104.csv')


# In[19]:


# Merging Failures and location datasets and fixing columns types (to have ssd failure data that has location, failure time, and smart att)
df_MergeDs_FailuresAppsLocation =  pd.merge(df_AlibabaOver_Failurelogs, df_AlibabaSnapShot_FailuresAppsLocation, how = 'inner', on = ['disk_id', 'failure_time'])

# Changing failure time column to datetime type
df_MergeDs_FailuresAppsLocation['failure_time'] =  pd.to_datetime(df_MergeDs_FailuresAppsLocation['failure_time'])

# Removing duplicates
#df_Failurelogs_FailuresAppsLocation = df_Failurelogs_FailuresAppsLocation.drop_duplicates(subset=['disk_id','failure_time'], keep="first")

# Forcing sorting
df_MergeDs_FailuresAppsLocation = df_MergeDs_FailuresAppsLocation.sort_values(by=['failure_time'], ascending=True)

# Choosing the columns of interest
df_MergeDs_FailuresAppsLocation = df_MergeDs_FailuresAppsLocation.loc[:,['disk_id','failure_time', 'model_x','model_y','app','node_id','rack_id','machine_room_id']]

# Changing data type
#df_Failurelogs_FailuresAppsLocation = df_Failurelogs_FailuresAppsLocation.astype(datatype)



# In[ ]:


## Testing (some queries)

#df_Failurelogs_FailuresAppsLocation.head(20)
#df_Failurelogs_FailuresAppsLocation.query(('rack_id == 17596 & app == "RM"'))
#df_Failurelogs_FailuresAppsLocation.query(('app == "RM"'))
#time1 = df_Failurelogs_FailuresAppsLocation.query(('rack_id == 17596 & app == "RM" & disk_id==39876'))
#time2 = df_Failurelogs_FailuresAppsLocation.query(('rack_id == 17596 & app == "RM" & disk_id==22968'))


# In[45]:


## Inserting OverTimeSSDsFailures into the DB

for row in df_AlibabaOver_Failurelogs.itertuples():
    insert_OverTimeSSDsFailures = OverTimeSSDsFailures()
    insert_OverTimeSSDsFailures.disk_id = row.disk_id
    insert_OverTimeSSDsFailures.failure_time = row.failure_time
    insert_OverTimeSSDsFailures.model_x = row.model
    insert_OverTimeSSDsFailures.save()


# In[46]:


## Inserting AllAppsSSDsFailures into the DB

for row in df_AlibabaSnapShot_FailuresAppsLocation.itertuples():
    insert_AllAppsSSDsFailures = AllAppsSSDsFailures()
    insert_AllAppsSSDsFailures.disk_id = row.disk_id
    insert_AllAppsSSDsFailures.app = row.app
    insert_AllAppsSSDsFailures.failure_time = row.failure_time
    insert_AllAppsSSDsFailures.node_id = row.node_id
    insert_AllAppsSSDsFailures.rack_id = row.rack_id
    insert_AllAppsSSDsFailures.machine_room_id = row.machine_room_id
    insert_AllAppsSSDsFailures.save()


# In[47]:


## Inserting AllAppsSSDsLocation into the DB

for row in df_AlibabaSnapShot_AllAppsSSDsLocation.itertuples():
    insert_AllAppsSSDsLocation = AllAppsSSDsLocation()
    insert_AllAppsSSDsLocation.disk_id = row.disk_id
    insert_AllAppsSSDsLocation.model_y = row.model
    insert_AllAppsSSDsLocation.app = row.app
    insert_AllAppsSSDsLocation.node_id = row.node_id
    insert_AllAppsSSDsLocation.rack_id = row.rack_id
    insert_AllAppsSSDsLocation.slot_id = row.slot_id
    insert_AllAppsSSDsLocation.save()


# In[51]:


## Inserting AllDiskIDSMARTAttributes into the DB

for row in df_AlibabaSnapShot_SMARTAttributes.itertuples():
    insert_AllDiskIDSMARTAttributes = AllDiskIDSMARTAttributes()
    insert_AllDiskIDSMARTAttributes.disk_id = row.disk_id
    insert_AllDiskIDSMARTAttributes.model_y = row.model
    insert_AllDiskIDSMARTAttributes.r_sectors = row.r_5
    insert_AllDiskIDSMARTAttributes.u_errors = row.n_187
    insert_AllDiskIDSMARTAttributes.p_on = row.r_9
    insert_AllDiskIDSMARTAttributes.p_c_count = row.r_12
    insert_AllDiskIDSMARTAttributes.p_failedA = row.n_171
    insert_AllDiskIDSMARTAttributes.p_failedB = row.n_181
    insert_AllDiskIDSMARTAttributes.e_failedA = row.n_172
    insert_AllDiskIDSMARTAttributes.e_failedB = row.n_182
    insert_AllDiskIDSMARTAttributes.n_b_written = row.r_241
    insert_AllDiskIDSMARTAttributes.n_b_read = row.r_242
    insert_AllDiskIDSMARTAttributes.w_l_count = row.r_173
    insert_AllDiskIDSMARTAttributes.w_r_d = row.r_177
    insert_AllDiskIDSMARTAttributes.media_wearout_i = row.n_233
    insert_AllDiskIDSMARTAttributes.save()


# In[52]:


## Inserting AllDiskIDSMARTAttributesFirstDayExperiment into the DB

for row in df_AlibabaSnapShot_SMARTAttributesFirstDay.itertuples():
    insert_AllDiskIDSMARTAttributesFirstDay = AllDiskIDSMARTAttributesFirstDay()
    insert_AllDiskIDSMARTAttributesFirstDay.disk_id = row.disk_id
    insert_AllDiskIDSMARTAttributesFirstDay.model_y = row.model
    insert_AllDiskIDSMARTAttributesFirstDay.r_sectors = row.r_5
    insert_AllDiskIDSMARTAttributesFirstDay.u_errors = row.n_187
    insert_AllDiskIDSMARTAttributesFirstDay.p_on = row.r_9
    insert_AllDiskIDSMARTAttributesFirstDay.p_c_count = row.r_12
    insert_AllDiskIDSMARTAttributesFirstDay.p_failedA = row.n_171
    insert_AllDiskIDSMARTAttributesFirstDay.p_failedB = row.n_181
    insert_AllDiskIDSMARTAttributesFirstDay.e_failedA = row.n_172
    insert_AllDiskIDSMARTAttributesFirstDay.e_failedB = row.n_182
    insert_AllDiskIDSMARTAttributesFirstDay.n_b_written = row.r_241
    insert_AllDiskIDSMARTAttributesFirstDay.n_b_read = row.r_242
    insert_AllDiskIDSMARTAttributesFirstDay.w_l_count = row.r_173
    insert_AllDiskIDSMARTAttributesFirstDay.w_r_d = row.r_177
    insert_AllDiskIDSMARTAttributesFirstDay.media_wearout_i = row.n_233
    insert_AllDiskIDSMARTAttributesFirstDay.save()


# In[20]:


## Inserting merged (disks and failures in common) FailuresAppsLocation into the DB

for row in df_MergeDs_FailuresAppsLocation.itertuples():
    insert_FailuresAppsLocation = FailuresAppsLocation()
    insert_FailuresAppsLocation.disk_id = row.disk_id
    insert_FailuresAppsLocation.failure_time = row.failure_time
    insert_FailuresAppsLocation.model_x = row.model_x
    insert_FailuresAppsLocation.model_y = row.model_y
    insert_FailuresAppsLocation.app = row.app
    insert_FailuresAppsLocation.node_id = row.node_id
    insert_FailuresAppsLocation.rack_id = row.rack_id
    insert_FailuresAppsLocation.machine_room_id = row.machine_room_id
    insert_FailuresAppsLocation.save()


# In[5]:


## Loading AlibabaOvertime dataset using Pandas and Inserting SMARTAtt into the DB 

start_date = date(2018, 4, 22)
end_date = date(2019, 12, 31)
delta = timedelta(days=1)
df_AlibabaOver_SMARTlogs = pd.DataFrame()


while start_date <= end_date:
    path = Path('/media/erb/hdd1/DataSet/alibabaOvertime/smartlogs/' + start_date.strftime("%Y%m%d") + '.csv')

    if path.is_file(): # checking if a particular file for a specific date is missing
        df_AlibabaOver_SMARTlogs = pd.read_csv(path)
        df_AlibabaOver_SMARTlogs = pd.DataFrame(df_AlibabaOver_SMARTlogs)

        # Changing failure time column to datetime type
        df_AlibabaOver_SMARTlogs['ds'] =  pd.to_datetime(df_AlibabaOver_SMARTlogs['ds'], format='%Y%m%d')

        # Choosing the columns of interest
        df_AlibabaOver_SMARTlogs = df_AlibabaOver_SMARTlogs.loc[:,['disk_id','ds', 'model','n_5','n_187','n_171','n_181','n_172','n_182','n_241','n_242']]

        # Changing the name of some columns to clarify their meaning
        df_AlibabaOver_SMARTlogs.rename(columns = {'ds':'timestamp', 'model':'model_x', 'n_5':'r_sectors','n_187':'u_errors','n_171':'p_failedA','n_181':'p_failedB','n_172':'e_failedA','n_182':'e_failedB','n_241':'n_b_written','n_242':'n_b_read'}, inplace=True)

        for row in df_AlibabaOver_SMARTlogs.itertuples():
            insert_SmartAttributes = SMARTAtt()
            insert_SmartAttributes.disk_id = row.disk_id
            insert_SmartAttributes.timestamp = row.timestamp
            insert_SmartAttributes.model_x = row.model_x

            #Checking if the value is int/float. Without this checking an error may be raised during mongoengine validation (saving)

            if isinstance(row.r_sectors, (int, float)):
                insert_SmartAttributes.r_sectors = row.r_sectors
            else: 
                insert_SmartAttributes.r_sectors = 0
            if isinstance(row.u_errors, (int, float)):
                insert_SmartAttributes.u_errors = row.u_errors
            else: 
                insert_SmartAttributes.u_errors = 0
            if isinstance(row.p_failedA, (int, float)):
                insert_SmartAttributes.p_failedA = row.p_failedA
            else: 
                insert_SmartAttributes.p_failedA = 0
            if isinstance(row.p_failedB, (int, float)):
                insert_SmartAttributes.p_failedB = row.p_failedB
            else: 
                insert_SmartAttributes.p_failedB = 0
            if isinstance(row.e_failedA, (int, float)):
                insert_SmartAttributes.e_failedA = row.e_failedA
            else: 
                insert_SmartAttributes.e_failedA = 0
            if isinstance(row.e_failedB, (int, float)):
                insert_SmartAttributes.e_failedB = row.e_failedB
            else: 
                insert_SmartAttributes.e_failedB = 0
            if isinstance(row.n_b_written, (int, float)):
                insert_SmartAttributes.n_b_written = row.n_b_written
            else: 
                insert_SmartAttributes.n_b_written = 0
            if isinstance(row.n_b_read, (int, float)):
                insert_SmartAttributes.n_b_read = row.n_b_read
            else: 
                insert_SmartAttributes.n_b_read = 0
            insert_SmartAttributes.save()

        #df_AlibabaOver_SMARTlogs_Filtered = pd.concat([df_AlibabaOver_SMARTlogs_Filtered, df_AlibabaOver_SMARTlogs], ignore_index=True)
    start_date += delta

# Changing the name of some columns to clarify their meaning
#df_AlibabaOver_SMARTlogs_Filtered.rename(columns = {'ds':'timestamp', 'model':'model_x', 'n_5':'r_sectors','n_187':'u_errors','n_171':'p_failedA','n_181':'p_failedB','n_172':'e_failedA','n_182':'e_failedB','n_241':'n_b_written','n_242':'n_b_read'}, inplace=True)


# In[56]:


## Test to perform the extraction of wear leveling values
# Loading AlibabaOvertime dataset using Pandas

start_date = date(2018, 1, 1)
end_date = date(2019, 12, 31)
delta = timedelta(days=1)
#df_AlibabaOver_SMARTlogs = pd.DataFrame()


while start_date <= end_date:
    path = Path('/media/erb/hdd1/DataSet/alibabaOvertime/smartlogs/' + start_date.strftime("%Y%m%d") + '.csv')

    if path.is_file(): # checking if a particular file for a specific date is missing
        df_AlibabaOver_SMARTlogsWear = pd.read_csv(path)
        df_AlibabaOver_SMARTlogsWear = pd.DataFrame(df_AlibabaOver_SMARTlogsWear)

        # Changing failure time column to datetime type
        df_AlibabaOver_SMARTlogsWear['ds'] =  pd.to_datetime(df_AlibabaOver_SMARTlogsWear['ds'], format='%Y%m%d')

        # Choosing the columns of interest
        df_AlibabaOver_SMARTlogsWear = df_AlibabaOver_SMARTlogsWear.loc[:,['disk_id','ds', 'model','r_5','r_173','r_177','r_233']]

        # Changing the name of some columns to clarify their meaning
        df_AlibabaOver_SMARTlogsWear.rename(columns = {'ds':'timestamp', 'model':'model_x', 'r_5':'r_sectors','r_173':'w_l_count','r_177':'w_r_d','r_233':'media_wearout_i'}, inplace=True)

        for row in df_AlibabaOver_SMARTlogsWear.itertuples():
            insert_SMARTAttOverWear = SMARTAttOverWear()
            insert_SMARTAttOverWear.disk_id = row.disk_id
            insert_SMARTAttOverWear.timestamp = row.timestamp
            insert_SMARTAttOverWear.model_x = row.model_x

            #Checking if the value is int/float. Without this checking an error may be raised during mongoengine validation (saving)

            if isinstance(row.r_sectors, (int, float)):
                insert_SMARTAttOverWear.r_sectors = row.r_sectors
            else: 
                insert_SMARTAttOverWear.r_sectors = 0
            if isinstance(row.w_l_count, (int, float)):
                insert_SMARTAttOverWear.w_l_count = row.w_l_count
            else: 
                insert_SMARTAttOverWear.w_l_count = 0
            if isinstance(row.w_r_d, (int, float)):
                insert_SMARTAttOverWear.w_r_d = row.w_r_d
            else: 
                insert_SMARTAttOverWear.w_r_d = 0
            if isinstance(row.media_wearout_i, (int, float)):
                insert_SMARTAttOverWear.media_wearout_i = row.media_wearout_i
            else: 
                insert_SMARTAttOverWear.media_wearout_i = 0
            insert_SMARTAttOverWear.save()

        #df_AlibabaOver_SMARTlogs_Filtered = pd.concat([df_AlibabaOver_SMARTlogs_Filtered, df_AlibabaOver_SMARTlogs], ignore_index=True)
    start_date += delta

# Changing the name of some columns to clarify their meaning
#df_AlibabaOver_SMARTlogs_Filtered.rename(columns = {'ds':'timestamp', 'model':'model_x', 'n_5':'r_sectors','n_187':'u_errors','n_171':'p_failedA','n_181':'p_failedB','n_172':'e_failedA','n_182':'e_failedB','n_241':'n_b_written','n_242':'n_b_read'}, inplace=True)


# In[ ]:


## Loading AlibabaOvertime dataset using Pandas and Inserting SMARTAtt into the DB 

start_date = date(2018, 1, 1)
end_date = date(2019, 12, 31)
delta = timedelta(days=1)
#df_AlibabaOver_SMARTlogs = pd.DataFrame()


while start_date <= end_date:
    path = Path('/root/alibabaOvertime/smartlogs/' + start_date.strftime("%Y%m%d") + '.csv')

    if path.is_file(): # checking if a particular file for a specific date is missing
        df_AlibabaOver_SMARTlogsFull = pd.read_csv(path)
        df_AlibabaOver_SMARTlogsFull = pd.DataFrame(df_AlibabaOver_SMARTlogsFull)

        # Changing failure time column to datetime type
        df_AlibabaOver_SMARTlogsFull['ds'] =  pd.to_datetime(df_AlibabaOver_SMARTlogsFull['ds'], format='%Y%m%d')

        # Choosing the columns of interest
        df_AlibabaOver_SMARTlogsFull = df_AlibabaOver_SMARTlogsFull.loc[:,['disk_id','ds', 'model','r_5', 'r_9','r_187','r_171','r_181','r_172','r_182','r_241','r_242','r_173','r_177','n_233']]
        
        # Changing the name of some columns to clarify their meaning
        df_AlibabaOver_SMARTlogsFull.rename(columns = {'ds':'timestamp', 'model':'model_x', 'r_5':'r_sectors', 'r_9':'power_hours' ,'r_187':'u_errors','r_171':'p_failedA','r_181':'p_failedB','r_172':'e_failedA','r_182':'e_failedB','r_241':'n_b_written','r_242':'n_b_read','r_173':'w_l_count','r_177':'w_r_d','n_233':'media_wearout_i'}, inplace=True)

        for row in df_AlibabaOver_SMARTlogsFull.itertuples():
            insert_SMARTAttFull = SMARTAttFull()
            insert_SMARTAttFull.disk_id = row.disk_id
            insert_SMARTAttFull.timestamp = row.timestamp
            insert_SMARTAttFull.model_x = row.model_x

            #Checking if the value is int/float. Without this checking an error may be raised during mongoengine validation (saving)

            if isinstance(row.r_sectors, (int, float)):
                insert_SMARTAttFull.r_sectors = row.r_sectors
            else: 
                insert_SMARTAttFull.r_sectors = 0
            if isinstance(row.power_hours, (int, float)):
                insert_SMARTAttFull.power_hours = row.power_hours
            else: 
                insert_SMARTAttFull.power_hours = 0
            if isinstance(row.u_errors, (int, float)):
                insert_SMARTAttFull.u_errors = row.u_errors
            else: 
                insert_SMARTAttFull.u_errors = 0
            if isinstance(row.p_failedA, (int, float)):
                insert_SMARTAttFull.p_failedA = row.p_failedA
            else: 
                insert_SMARTAttFull.p_failedA = 0
            if isinstance(row.p_failedB, (int, float)):
                insert_SMARTAttFull.p_failedB = row.p_failedB
            else: 
                insert_SMARTAttFull.p_failedB = 0
            if isinstance(row.e_failedA, (int, float)):
                insert_SMARTAttFull.e_failedA = row.e_failedA
            else: 
                insert_SMARTAttFull.e_failedA = 0
            if isinstance(row.e_failedB, (int, float)):
                insert_SMARTAttFull.e_failedB = row.e_failedB
            else: 
                insert_SMARTAttFull.e_failedB = 0
            if isinstance(row.n_b_written, (int, float)):
                insert_SMARTAttFull.n_b_written = row.n_b_written
            else: 
                insert_SMARTAttFull.n_b_written = 0
            if isinstance(row.n_b_read, (int, float)):
                insert_SMARTAttFull.n_b_read = row.n_b_read
            else: 
                insert_SMARTAttFull.n_b_read = 0
            if isinstance(row.w_l_count, (int, float)):
                insert_SMARTAttFull.w_l_count = row.w_l_count
            else: 
                insert_SMARTAttFull.w_l_count = 0
            if isinstance(row.w_r_d, (int, float)):
                insert_SMARTAttFull.w_r_d = row.w_r_d
            else: 
                insert_SMARTAttFull.w_r_d = 0
            if isinstance(row.media_wearout_i, (int, float)):
                insert_SMARTAttFull.media_wearout_i = row.media_wearout_i
            else: 
                insert_SMARTAttFull.media_wearout_i = 0
            insert_SMARTAttFull.save()

        #df_AlibabaOver_SMARTlogs_Filtered = pd.concat([df_AlibabaOver_SMARTlogs_Filtered, df_AlibabaOver_SMARTlogs], ignore_index=True)
    start_date += delta


# In[ ]:


## Loading Backblaze dataset using Pandas and Inserting SMARTAttFullBackBlaze into the DB 

start_date = date(2018, 1, 1)
end_date = date(2022, 6, 30)
delta = timedelta(days=1)

while start_date <= end_date:
    path = Path('/root/backblaze/' + start_date.strftime("%Y-%m-%d") + '.csv')

    if path.is_file(): # checking if a particular file for a specific date is missing
        df_SMARTAttFullBackBlaze = pd.read_csv(path)
        df_SMARTAttFullBackBlaze = pd.DataFrame(df_SMARTAttFullBackBlaze)

        # Changing failure time column to datetime type
        df_SMARTAttFullBackBlaze['date'] =  pd.to_datetime(df_SMARTAttFullBackBlaze['date'], format='%Y-%m-%d')

        # Choosing the columns of interest
        df_SMARTAttFullBackBlaze = df_SMARTAttFullBackBlaze.loc[:,['serial_number', 'model', 'date', 'capacity_bytes', 'failure', 'smart_5_raw', 'smart_9_raw','smart_187_raw','smart_1_raw','smart_241_raw','smart_242_raw','smart_188_raw','smart_197_raw','smart_198_raw']]
        
        # Changing the name of some columns to clarify their meaning
        df_SMARTAttFullBackBlaze.rename(columns = {'serial_number':'disk_id', 'date':'timestamp', 'smart_5_raw':'r_sectors', 'smart_9_raw':'power_hours' ,'smart_187_raw':'u_errors','smart_1_raw':'read_error_rate','smart_241_raw':'n_b_written','smart_242_raw':'n_b_read','smart_188_raw':'command_timeout','smart_197_raw':'current_pending_sector_count','smart_198_raw':'uncorrectable_sector_count'}, inplace=True)

        for row in df_SMARTAttFullBackBlaze.itertuples():
            insert_SMARTAttFullBackBlaze = SMARTAttFullBackBlaze()
            insert_SMARTAttFullBackBlaze.disk_id = row.disk_id
            insert_SMARTAttFullBackBlaze.timestamp = row.timestamp
            insert_SMARTAttFullBackBlaze.model = row.model

            #Checking if the value is int/float. Without this checking an error may be raised during mongoengine validation (saving)
            if isinstance(row.capacity_bytes, (int, float)):
                insert_SMARTAttFullBackBlaze.capacity_bytes = row.capacity_bytes
            else: 
                insert_SMARTAttFullBackBlaze.capacity_bytes = 0
            if isinstance(row.failure, (int, float)):
                insert_SMARTAttFullBackBlaze.failure = row.failure
            else: 
                insert_SMARTAttFullBackBlaze.failure = 0            
            if isinstance(row.r_sectors, (int, float)):
                insert_SMARTAttFullBackBlaze.r_sectors = row.r_sectors
            else: 
                insert_SMARTAttFullBackBlaze.r_sectors = 0
            if isinstance(row.power_hours, (int, float)):
                insert_SMARTAttFullBackBlaze.power_hours = row.power_hours
            else: 
                insert_SMARTAttFullBackBlaze.power_hours = 0
            if isinstance(row.u_errors, (int, float)):
                insert_SMARTAttFullBackBlaze.u_errors = row.u_errors
            else: 
                insert_SMARTAttFullBackBlaze.u_errors = 0
            if isinstance(row.read_error_rate, (int, float)):
                insert_SMARTAttFullBackBlaze.read_error_rate = row.read_error_rate
            else: 
                insert_SMARTAttFullBackBlaze.read_error_rate = 0
            if isinstance(row.n_b_written, (int, float)):
                insert_SMARTAttFullBackBlaze.n_b_written = row.n_b_written
            else: 
                insert_SMARTAttFullBackBlaze.n_b_written = 0
            if isinstance(row.n_b_read, (int, float)):
                insert_SMARTAttFullBackBlaze.n_b_read = row.n_b_read
            else: 
                insert_SMARTAttFullBackBlaze.n_b_read = 0
            if isinstance(row.command_timeout, (int, float)):
                insert_SMARTAttFullBackBlaze.command_timeout = row.command_timeout
            else: 
                insert_SMARTAttFullBackBlaze.command_timeout = 0
            if isinstance(row.current_pending_sector_count, (int, float)):
                insert_SMARTAttFullBackBlaze.current_pending_sector_count = row.current_pending_sector_count
            else: 
                insert_SMARTAttFullBackBlaze.current_pending_sector_count = 0
            if isinstance(row.uncorrectable_sector_count, (int, float)):
                insert_SMARTAttFullBackBlaze.uncorrectable_sector_count = row.uncorrectable_sector_count
            else: 
                insert_SMARTAttFullBackBlaze.uncorrectable_sector_count = 0
            insert_SMARTAttFullBackBlaze.save()
    start_date += delta

