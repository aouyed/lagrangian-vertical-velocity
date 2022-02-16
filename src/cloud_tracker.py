#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:57:26 2022

@author: aouyed
"""

import cv2
import numpy as np
import xarray as xr
import main as m 
import calculators as c
import config as config 
from centroidtracker import CentroidTracker
import main as m 


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
    #thresh = cv2.dilate(thresh, kernel, iterations=1)    

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
        


def main():
    #ds_total=object_tracker()
    ds_total=xr.open_dataset('../data/processed/tracked.nc')
    print(ds_total)
    cmap = c.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    ds_total=ds_total.sel(lat=slice(21,23), lon=slice(-91,-88))
    #ds_total=ds_total.sel(lat=slice(19,30))
    ds_total['size_map']=np.sqrt(ds_total.area_map)
    ds_total['pressure_vel']=100*ds_total['pressure_vel']
    ds_total['pressure_tendency']=100*ds_total['pressure_tendency']
    
    time=ds_total['time'].values[1]
    ds=ds_total.sel(time=time)
    ids=ds['id_map'].values
    ids, id_counts=np.unique(ids, return_counts=True)
    print(ids)
    print(id_counts)
    ds_total=ds_total.where(ds_total.id_map==ids[3])

    m.plot_loop(ds_total, 'thresh_map', c.implot, 0, 255,'viridis',m.FOLDER)
    m.plot_loop(ds_total, 'id_map', c.implot, 0, 1000,cmap,m.FOLDER)
    m.plot_loop(ds_total, 'size_map', c.implot, 0, 100,'viridis',m.FOLDER)
    m.plot_loop(ds_total, 'pressure_vel', c.implot, -2, 2,'RdBu',m.FOLDER)
    m.plot_loop(ds_total, 'pressure_tendency', c.implot, -2, 2,'RdBu',m.FOLDER)
    m.plot_loop(ds_total, 'cloud_top_pressure', c.implot, 0, 1000,'viridis',m.FOLDER)

    print(ds_total)
    
if __name__ == "__main__":
    main()
