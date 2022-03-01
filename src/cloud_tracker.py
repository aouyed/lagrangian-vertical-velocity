#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:57:26 2022

@author: aouyed
"""

import cv2
import numpy as np
import xarray as xr
import pandas as pd
import main as m 
import calculators as c
import kinematics_tracker as kt
import config as config 
from centroidtracker import CentroidTracker
import main as m 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 
from scipy import stats

IDS=[113, 114, 115, 117, 118, 120, 125, 131]

def contour_filter(contours):
    dummy_c=[]
    for contour in contours:
        M=cv2.moments(contour)
        if M['m00'] >0:
            dummy_c.append(contour)
            
    contours=dummy_c
    return contours
    
def countourer(values, kernelv, threshv,  area_lim=100):
    values = cv2.blur(values, (10, 10))

    _, thresh = cv2.threshold(values.copy(), threshv, 255,cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    kernel = np.ones((kernelv, kernelv), np.uint8)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours=contour_filter(contours)
    return contours

def contour_plotter(ds_clouds,new_cmap, i):
    ds_clouds=ds_clouds.sel(lat=slice(10,25), lon=slice(-91,-80))
    ids=ds_clouds.id_map.values
    ids=np.nan_to_num(ids)
    c.map_plotter(ds_clouds, 'cloud_top_pressure_'+str(i),'cloud_top_pressure', units_label='hpa')
    c.map_plotter(ds_clouds, 'contours_id_'+str(i),'id_map',cmap=new_cmap, vmin=0,vmax=1000)
    c.map_plotter(ds_clouds, 'contours_text'+str(i),'text_map')


def contour_store(contours_dict,date, contours ):
    for (objectID, contour) in contours.items():
        contours_dict['id'].append(objectID)
        contours_dict['contour'].append(contour)
        contours_dict['date'].append(date)
        

def contour_drawer(contours, objects, ds_clouds):
    values =ds_clouds['cloud_top_pressure'].values
    values = np.squeeze(values)  
    id_map=np.zeros_like(values)
    text_map=np.zeros_like(values)
    area_map=np.zeros_like(values)

    for (objectID, contour) in contours.items():
            cv2.drawContours(id_map, [contour], -1, objectID, -1)
            area = cv2.contourArea(contour)
            cv2.drawContours(area_map, [contour], -1, area, -1)

            
           
    for (objectID, centroid) in objects.items():
         text = str(objectID)
         cv2.putText(text_map, text, (centroid[0], centroid[1]),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.25, 255, 1, False)
    threshv = config.THRESH
    _, thresh = cv2.threshold(values.copy(), threshv, 255,cv2.THRESH_BINARY)
    ds_clouds['thresh_map']=(['lat','lon'], thresh)

    ds_clouds['id_map']=(['lat','lon'], id_map)
    ds_clouds['text_map']=(['lat','lon'], text_map)
    ds_clouds['area_map']=(['lat','lon'],area_map)

    return ds_clouds

def contour_loop(ds):
    kernel = config.KERNEL
    threshv = config.THRESH
    ct = CentroidTracker()
    ds_total=xr.Dataset()
    contours_dict={'date': [], 'id':[], 'contour':[]}
    new_cmap = c.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)

    for i, time in  enumerate(ds['time'].values):
        print(time)
        ds_clouds= ds[['cloud_top_pressure','pressure_vel','pressure_tendency','flow_x','flow_y']].sel(time=time)
        ds_clouds=ds_clouds.fillna(0)
        values =ds_clouds['cloud_top_pressure'].values
        values = np.squeeze(values)    
        contours= countourer(values, kernel, threshv)
        objects, contours = ct.update(contours)
        ds_clouds= contour_drawer(contours, objects,ds_clouds)
        contour_plotter(ds_clouds, new_cmap, i)
        contour_store(contours_dict, time, contours )
        
        
        if ds_total:
            ds_total=xr.concat([ds_total,ds_clouds],'time' )
        else:
            ds_total=ds_clouds
    
    pickle.dump(contours_dict,open( "../data/processed/tracked_contours_dictionary.p", "wb" ))
    df=pd.DataFrame(data=contours_dict)
    df=df.set_index(['date','id'])
    ds_contours=xr.Dataset.from_dataframe(df)

    return ds_total, new_cmap, ds_contours, df
       

def object_tracker():
    file=   m.NC_PATH+m.FOLDER+'_output.nc'
    ds=xr.open_dataset(file)
    ds_total, cmap, ds_contours, df =contour_loop(ds)  
    ds_total.to_netcdf('../data/processed/tracked.nc')
    pickle.dump(ds_contours,open( "../data/processed/tracked_contours.p", "wb" ))
    df.to_pickle('../data/processed/tracked_contours_df.p')
   

    return ds_total, ds_contours, df 


def time_loop(ds_total, idno,labels, data,  median):

    for time in ds_total['time'].values:
        ds=ds_total.sel(time = time)     
        for label in labels:
            if median:
                values=ds[label].median().item()
                data[label].append(values)
                #pressure_vel=ds['pressure_vel'].median().item()
                #pressure_t=ds['pressure_tendency'].median().item()
                #pressure=ds['cloud_top_pressure'].median().item()
            else:
               values=ds[label].mean().item()
               data[label].append(values)
            
        data['time'].append(time)
        data['id'].append(idno) 
    return data


def time_series_calc(ds_total, ids,labels,  median=False):
    #data={'time':[], 'id':[],'size':[],'pressure_vel':[],'pressure_ten':[],'cloud_top_pressure':[]}
    data={}
    for label in labels:
        data[label]=[]
    data['time']=[]
    data['id']=[]

    for idno in tqdm(ids[ids!=0]): 
        ds=ds_total.where(ds_total.id_map==idno)
        data=time_loop( ds,idno,labels,data,  median)
            
            
        
    df=pd.DataFrame(data)
    df=df.set_index(['time','id'])
    ds_time_series=xr.Dataset.from_dataframe(df)
    stats='median'
    if  not median:
        stats='mean'
  
    ds_time_series.to_netcdf('../data/processed/time_series_complete_'+stats+'.nc')
    
    return ds_time_series    


        

def time_series_plotter(ds, label):
    fig, ax= plt.subplots()
    #for idno in ds['id'].values:
    for idno in ds['id'].values:
        ds_unit=ds.sel(id=idno)
        ax.plot(ds_unit['time'].values,ds_unit[label])
    ax.legend()
    #ax.set_ylim(0,150)
    plt.savefig('../data/processed/plots/ts_'+label+'.png', dpi=300)
    plt.show()
    plt.close()
    
    
def scatter_plot(ds, label):
    df=ds.to_dataframe().reset_index()
    df=df.dropna()
    means_vel, edges, _=stats.binned_statistic(df['size_rate'], df['pressure_vel'], 'mean', bins=10, range=(-10,10))
    means_ten, edges, _=stats.binned_statistic(df['size_rate'], df['pressure_ten'], 'mean', bins=10, range=(-10,10))
    means_rate, edges, _=stats.binned_statistic(df['size_rate'], df['pressure_rate'], 'mean', bins=10, range=(-10,10))

    edges=edges[1:]
    fig, ax= plt.subplots()
    #ax.scatter(df[label],df['pressure_rate'])
    ax.plot(edges, means_vel)
    ax.plot(edges,means_ten)
    ax.plot(edges,means_rate)
    plt.show()
    plt.close()
    
    
def scatter_general(ds, labelx, labely):
    df=ds.to_dataframe().reset_index()
    fig, ax= plt.subplots()
    ax.scatter(df[labelx],df[labely])
    plt.show()
    plt.close()
    
def time_series(ds_total, median=False):
    ds_total['size_map']=np.sqrt(ds_total.area_map)
    ds_total['pressure_vel']=100*ds_total['pressure_vel']
    ds_total['pressure_tendency']=100*ds_total['pressure_tendency']
    ids=np.unique(ds_total['id_map'].values)

    ds_time_series=time_series_calc(ds_total, ids, median)
    stats='median'
    
    
   

def cloud_plotter(da, vmin=-1, vmax=1, cmap='RdBu'):
    if vmin!=vmax:
        da.plot(vmin=-1, vmax=1, cmap=cmap)
    else:
        da.plot(cmap=cmap)
    plt.show()
    plt.close()


def mean_clouds(ds_time_series,ds_cloud, ds_contours,labels, string='mean'):
    #labels=['pressure_vel','pressure_ten','cloud_top_pressure','pressure_adv','pressure_rate','size_rate']
    ds_total=xr.Dataset()
    for date in tqdm(ds_contours['date'].values):
        image_di={}
        ds_unit=ds_cloud.sel(time=date)
        for label in labels:
            image_di[label]=np.zeros_like(ds_unit['cloud_top_pressure'].values)
        for idno in ds_contours['id'].values[ds_contours['id'].values>0]:
            da=ds_contours.sel(date=date, id=idno)
            contour=da['contour'].values
            contour=np.squeeze(contour).item()
            ds_time=ds_time_series.sel(id=idno, time=date)   
            try:
                for label in image_di:
                    cv2.drawContours(image_di[label], [contour], -1,ds_time[label].values.item(), -1)

            except:
                print("bad contour value:")
                print(contour)
        for  label in image_di:
            img=image_di[label]
            img[img==0]=np.nan
            ds_unit[label+'_mean']=(['lat','lon'], img)
        ds_unit=ds_unit.expand_dims('time')
        if not ds_total:
            ds_total=ds_unit
        else:
            ds_total=xr.concat([ds_total, ds_unit], 'time')
    ds_total.to_netcdf('../data/processed/clouds_'+string+'.nc')
    return ds_total

    
    
def animation(ds_total, tag):
    cmap = c.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)

    m.plot_loop(ds_total, 'id_map', c.implot, 0, 1000,cmap,m.FOLDER + tag)
    m.plot_loop(ds_total, 'size_rate_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
    m.plot_loop(ds_total, 'pressure_adv_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
    m.plot_loop(ds_total, 'cloud_top_pressure_mean', c.implot, 0, 1000,'viridis',m.FOLDER+tag)
    m.plot_loop(ds_total, 'pressure_vel_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
    m.plot_loop(ds_total, 'pressure_ten_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
    m.plot_loop(ds_total, 'pressure_rate_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+ tag)
    m.plot_loop(ds_total, 'thresh_map', c.implot, 0, 255,'viridis',m.FOLDER + tag)
    m.plot_loop(ds_total, 'size_map', c.implot, 0, 100,'viridis',m.FOLDER + tag)
    m.plot_loop(ds_total, 'pressure_vel', c.implot, -2, 2,'RdBu',m.FOLDER + tag)
    m.plot_loop(ds_total, 'pressure_tendency', c.implot, -2, 2,'RdBu',m.FOLDER + tag)
    m.plot_loop(ds_total, 'cloud_top_pressure', c.implot, 0, 1000,'viridis',m.FOLDER + tag)


def time_series_post_processing(stats):
    ds_time_series=xr.open_dataset('../data/processed/time_series_complete_'+stats+'.nc')
    ds_time_series=ds_time_series.rolling(time=3, center=True).mean()
    ds_time_series['size_rate']=ds_time_series['size'].differentiate("time",datetime_unit='h' )
    ds_time_series['pressure_rate']=100*ds_time_series['cloud_top_pressure'].differentiate("time",datetime_unit='s' )
    #ds_time_series=ds_time_series.where(ds_time_series.size_rate)
    ds_time_series['pressure_adv']=ds_time_series['pressure_vel'] -ds_time_series['pressure_ten']
    return ds_time_series
    
def post_processing():
    ds_total=xr.open_dataset('../data/processed/tracked.nc')
    ds_contours=pickle.load(open( "../data/processed/tracked_contours.p", "rb" ))
    df=pd.read_pickle('tracked_contours_df.p')
    ds_total['size_map']=np.sqrt(ds_total.area_map)
    ds_total['pressure_vel']=100*ds_total['pressure_vel']
    ds_total['pressure_tendency']=100*ds_total['pressure_tendency']
    ds_total['pressure_adv']= ds_total['pressure_vel']- ds_total['pressure_tendency']


    ds_time_series_mean= time_series_post_processing('mean')
    ds_time_series_median= time_series_post_processing('median')


    #time_series_plotter(ds_time_series, 'pressure_vel')
    #time_series_plotter(ds_time_series, 'size')
    #time_series_plotter(ds_time_series, 'size_rate')
    #time_series_plotter(ds_time_series, 'pressure_vel')
    #scatter_plot(ds_time_series, 'size_rate')
    #analyzer(ds_time_series, ds_total)
    mean_clouds(ds_time_series_mean, ds_total, ds_contours, 'mean')
    mean_clouds(ds_time_series_median, ds_total, ds_contours, 'median')

   

def main():
    ds_total=xr.open_dataset('../data/processed/clouds_mean_kin.nc')

    #ds_total=kt.calc(ds_total)
    
    ds_total['size_map']=np.sqrt(ds_total.area_map)
    ds_total['pressure_vel']=100*ds_total['pressure_vel']
    ds_total['pressure_tendency']=100*ds_total['pressure_tendency']
    ids=np.unique(ds_total['id_map'].values)
    labels=['pressure_vel','size_map','pressure_tendency','cloud_top_pressure','vorticity','divergence']
    ds_time_series=time_series_calc(ds_total, ids,labels,  median=False)
  
    ds_total=ds_total.sel(lat=slice(0,23))
    ids=np.unique(ds_total['id_map'].values)
    ds_time_series= time_series_post_processing('mean')
    scatter_plot(ds_time_series, 'size')
    scatter_plot(ds_time_series, 'cloud_top_pressure')  
    #ds_total
    ds_time_series=ds_time_series.sel(id=ids, method='nearest')
    time_series_plotter(ds_time_series, 'cloud_top_pressure')
    time_series_plotter(ds_time_series, 'pressure_vel')
    time_series_plotter(ds_time_series, 'pressure_rate')
    time_series_plotter(ds_time_series, 'size')

    scatter_plot(ds_time_series, 'size')
    scatter_plot(ds_time_series, 'cloud_top_pressure')  
    scatter_plot(ds_time_series, 'size_rate')
    #scatter_general(ds_time_series,'id', 'size_rate')
    #scatter_general(ds_time_series,'id', 'pressure_vel')  
    #scatter_general(ds_time_series,'id', 'pressure_rate')   

 
    #ds_unit=ds_total.where(np.sign(ds_total.pressure_ten_mean) != 
                            #np.sign(ds_total.pressure_vel_mean))
    #animation(ds_unit, '_opposite_signs')
    #ds_unit=ds_total.where(ds_total.id_map.isin(IDS))
   # animation(ds_total, '_list_tropics')


    # ds_total=xr.open_dataset('../data/processed/tracked.nc')
    # #time_series(ds_total, median=True)
    # #time_series(ds_total, median=False)
    # #ds_total, ds_contours, df=object_tracker()
    # post_processing()
    #ds_time_series= time_series_post_processing('mean')
    # condition1=np.sign(ds_time_series.pressure_ten) != np.sign(ds_time_series.pressure_vel)
    # condition2= ds_time_series.pressure_vel>0
                
    # ds_time_series=ds_time_series.where(condition1)
    # time_series_plotter(ds_time_series, 'pressure_vel')
    # print(ds_time_series['cloud_top_pressure'].mean())
    # print(ds_time_series['pressure_vel'].mean())
    # print(ds_time_series['pressure_ten'].mean())
    # print(ds_time_series['pressure_rate'].mean())




    


   
    
if __name__ == "__main__":
    main()
