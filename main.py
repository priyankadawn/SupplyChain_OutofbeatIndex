# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:16:52 2021

@author: Priyanka
"""

import scipy
from scipy import spatial
import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt
import datetime
import os
import pandas as pd
import plotly
import plotly.express as px

#setting the path for saving csv files if required
data_path = './Data'


#Writing the Input csv files
Lat_long=pd.read_csv(os.path.join(data_path,"Input_lattitude_longitude_sample.csv"))

Days_visit=pd.read_csv(os.path.join(data_path,"Input_days_visit_sample.csv"))

df = Lat_long
fig = px.scatter(df, x="lattitude", y="longitude", color="lattitude",
                 size='lattitude', hover_data=['lattitude'])
fig.show()


#Creating the probability distribution for assigning the probability as per distance
#Change this to automatically splitting it into distance buckets

Beat_Prob_master = pd.DataFrame({'Distance_limit':\
                                 [0,0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,3,1000],\
                                 'Prob_asigned':[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]})
Beat_Prob_master.sort_values(['Distance_limit'],inplace = True)


##Processing the days visit file
Days_visit_processed = Days_visit.replace(r'^\s*$', np.nan, regex=True)
Days_visit_processed=Days_visit_processed[Days_visit_processed['Service_days'].notnull()]
##only considering active outlets
Days_visit_processed=Days_visit_processed[Days_visit_processed['par_status'] =='ACTIVE']
Days_visit_processed.sort_values(['Distributor_code', 'Store_code', 'Partner_code', 'Updated_timestamp'],inplace =True)
Days_visit_processed['Updated_timestamp'] = pd.to_datetime(Days_visit_processed['Updated_timestamp'])

Days_visit_processed_input = Days_visit_processed[['Distributor_code', \
                                                   'Store_code','Service_days']]\
.groupby(['Distributor_code', 'Store_code'], as_index=False)\
.agg(lambda x: ','.join(set(x)))


Days_visit_processed_input=Days_visit_processed_input.\
applymap(lambda Service_days:','.join(set(Service_days.split(','))))

##Processing the lat long file

Lat_long = Lat_long.replace(r'^\s*$', np.nan, regex=True)
Lat_long=Lat_long[Lat_long['lattitude'].notnull()]

##Merging the service days and lat,long in final data
Merged_data = Lat_long.merge(Days_visit_processed_input,how = 'left',on=['Distributor_code', 'Store_code'])
#del Merged_data['nunique']

##Haversine distance function using 
def get_distance(point1, point2):
    R = 6371
    lat1 = radians(point1[0])
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    dlon = lon2 - lon1
    dlat = lat2- lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    distance = R * c
    return distance

##Getting the RS list
rs_list = Days_visit_processed_input['Distributor_code'].unique()
##Creating the output dataframe
OOB_prob_weekly = pd.DataFrame()

rs_data = Merged_data[Merged_data['Distributor_code']==Distributor_name]
rs_data['lattitude']=rs_data['lattitude'].astype('float')
rs_data['longitude']=rs_data['longitude'].astype('float')
##exluding no latitude stores
rs_data=rs_data[rs_data['lattitude'].notnull()]
##exluding no day of visit stores
rs_data=rs_data[rs_data['Service_days'].notnull()]

days_to_visit = pd.DataFrame(rs_data['Service_days'].apply(lambda x: pd.Series(str(x).split(',')))).rename(columns={0:"Day1",1:"Day2",2:"Day3",3:"Day4",4:"Day5",5:"Day6",6:"Day7"})
##Getting the day of visit dummies


days_to_visit=days_to_visit.dropna(axis=1,how='all')

Day_list=['M','T','W','Th','F','S','Su']
Day_dict={'M':'Monday','T':'Tuesday','W':'Wednesday','Th':'Thursday','F':'Friday','S':'Saturday','Su':'Sunday'}

daysofvisit_agg=pd.DataFrame()
for i in Day_list:
    daysofvisit_agg[i]=days_to_visit.loc[:, [x for x in days_to_visit.columns if x.endswith(i)]].count(axis=1)


##Creating the dataframe with the day visited
rs_days=pd.concat([rs_data['Store_code'],daysofvisit_agg],axis=1)

##Creating the haversine distance matrix
all_points = rs_data[['lattitude', 'longitude']].values
distancematrix = scipy.spatial.distance.cdist(all_points,all_points,get_distance)
Distance_matrix_rs=pd.DataFrame(distancematrix, index=rs_data['Store_code'], columns=rs_data['Store_code'])

##Result Summarization
rs_index2=pd.DataFrame()
result=pd.DataFrame()
weekly_index=pd.DataFrame()

day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for rs in Outlet_probability_week.RS.unique():
    df=Outlet_probability_week[Outlet_probability_week['RS']==rs]
    df=df[['Day','Prob_cutoffs']]
    result=pd.pivot_table(df,columns=["Prob_cutoffs"],index=["Day"], aggfunc = len).fillna(0)
    result1= result.apply(lambda x: x / result.sum(axis=1))
    ##sum product
    rs_index_day =  pd.DataFrame((result1 * result1.columns).sum(axis=1)).rename(columns={0:'Weekly_index'})
    rs_index_day['RS'] = rs
    rs_index_day['Day1'] =rs_index_day.index
    #Weekly RS INDEX
    weekly_index=weekly_index.append(rs_index_day)
    weekly_index['Day1'] = pd.Categorical(weekly_index['Day1'] , categories=day_name, ordered=True)
    weekly_index.sort_values(['RS','Day1'],inplace = True)
    rs_index = pd.DataFrame({'RS':[rs],'Rs_index':[rs_index_day['Weekly_index'].mean()]})
    ##RS INDEX 
    rs_index2=rs_index2.append(rs_index)
    rs_index2.sort_values(['RS'],inplace = True)
    
    
#print(weekly_index)
print(rs_index2)  

##Writing out the output files
weekly_index.to_csv('Weekly_RS_Index.csv',index = False)
rs_index2.to_csv('Outlet_RS_Index.csv',index = False)
