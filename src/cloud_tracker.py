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
import config as config 
from centroidtracker import CentroidTracker
import main as m 
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        
        if ds_total:
            ds_total=xr.concat([ds_total,ds_clouds],'time' )
        else:
            ds_total=ds_clouds
    
    return ds_total, new_cmap
       

def object_tracker():
    file=   m.NC_PATH+m.FOLDER+'_output.nc'
    ds=xr.open_dataset(file)
    ds_total, cmap =contour_loop(ds)  
    ds_total.to_netcdf('../data/processed/tracked.nc')
    #m.plot_loop(ds_total, 'id_map', c.implot, 0, 1000,cmap,m.FOLDER)
   

    return ds_total


def time_loop(data, ds_total, idno):
    for time in ds_total['time'].values:
        ds=ds_total.sel(time = time)     
        area=ds['size_map'].median().item()
        pressure_vel=ds['pressure_vel'].median().item()
        pressure_t=ds['pressure_tendency'].median().item()
    
        pressure=ds['cloud_top_pressure'].median().item()
        data['time'].append(time)
        data['id'].append(idno)
        data['size'].append(area)
        data['pressure_vel'].append(pressure_vel)
        data['pressure_ten'].append(pressure_t)
    
        data['cloud_top_pressure'].append(pressure)
    return data


def time_series(ds_total, ids):
    data={'time':[], 'id':[],'size':[],'pressure_vel':[],'pressure_ten':[],'cloud_top_pressure':[]}

    for idno in tqdm(ids[ids!=0]): 
        ds=ds_total.where(ds_total.id_map==idno)
        data=time_loop(data, ds, idno)
            
            
        
    df=pd.DataFrame(data)
    df=df.set_index(['time','id'])
    ds_time_series=xr.Dataset.from_dataframe(df)
    
    return ds_time_series    
        

def time_series_plotter(ds, label):
    fig, ax= plt.subplots()
    for idno in ds['id'].values:
        if idno==137:
            ds_unit=ds.sel(id=idno)
            ax.plot(ds_unit['time'].values,ds_unit[label],label= str(idno))
    ax.legend()
    #ax.set_ylim(0,150)
    plt.show()
    plt.close()
    
    
def scatter_plot(ds, label):
    df=ds.to_dataframe().reset_index()
    fig, ax= plt.subplots()
    ax.scatter(df[label],df['pressure_vel'])
    ax.scatter(df[label],df['pressure_ten'])
    
    if label is 'size':
        ax.set_xlim(0,200)

    plt.show()
    plt.close()
    
    
def scatter_general(ds, labelx, labely):
    df=ds.to_dataframe().reset_index()
    fig, ax= plt.subplots()
    ax.scatter(df[labelx],df[labely])
    plt.show()
    plt.close()
    
    
def main():
    #ds_total=object_tracker()
    ds_total=xr.open_dataset('../data/processed/tracked.nc')
    print(ds_total)
    
    ds_total['size_map']=np.sqrt(ds_total.area_map)
    ds_total['pressure_vel']=100*ds_total['pressure_vel']
    ds_total['pressure_tendency']=100*ds_total['pressure_tendency']
    ids=np.unique(ds_total['id_map'].values)

    ds_time_series=time_series(ds_total, ids)
    ds_time_series.to_netcdf('../data/processed/time_series_complete_median.nc')
    ds_time_series=xr.open_dataset('../data/processed/time_series_complete_median.nc')
    ds_time_series=ds_time_series.rolling(time=3, center=True).mean()
    ds_time_series['size_rate']=ds_time_series['size'].differentiate("time",datetime_unit='h' )

    #time_series_plotter(ds_time_series, 'pressure_vel')
    #time_series_plotter(ds_time_series, 'size')
    #time_series_plotter(ds_time_series, 'size_rate')
    #time_series_plotter(ds_time_series, 'pressure_vel')
    scatter_plot(ds_time_series, 'size_rate')
    scatter_plot(ds_time_series, 'size')
    scatter_plot(ds_time_series, 'cloud_top_pressure')
    scatter_general(ds_time_series, 'size', 'size_rate')

    breakpoint()

    m.plot_loop(ds_total, 'thresh_map', c.implot, 0, 255,'viridis',m.FOLDER)
    m.plot_loop(ds_total, 'id_map', c.implot, 0, 1000,cmap,m.FOLDER)
    m.plot_loop(ds_total, 'size_map', c.implot, 0, 100,'viridis',m.FOLDER)
    m.plot_loop(ds_total, 'pressure_vel', c.implot, -2, 2,'RdBu',m.FOLDER)
    m.plot_loop(ds_total, 'pressure_tendency', c.implot, -2, 2,'RdBu',m.FOLDER)
    m.plot_loop(ds_total, 'cloud_top_pressure', c.implot, 0, 1000,'viridis',m.FOLDER)

    print(ds_total)
    
if __name__ == "__main__":
    main()
