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
from centroidtracker import CentroidTracker
from PIL import Image, ImageFont, ImageDraw




def countourer(values, kernelv, threshv,  area_lim=100):
    values = cv2.blur(values, (10, 10))

    _, thresh = cv2.threshold(values.copy(), threshv, 255,cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    kernel = np.ones((kernelv, kernelv), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)    #thresh=cv2.blur(thresh,(8,8))
    #thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    dummy_c=[]
    for contour in contours:
        M=cv2.moments(contour)
        if M['m00'] >0:
            dummy_c.append(contour)
            
    contours=dummy_c
    # print(np.shape(thresh))
    contours_map = np.zeros(thresh.shape)
    track_map = np.zeros(thresh.shape)
    areas_map = np.zeros(thresh.shape)
    areas = []
    contoursm = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        length = np.sqrt(area)
        if area > area_lim:
            n = i
            # cv2.drawContours(zeros, contour, -1, 255, 10)
            cv2.drawContours(contours_map, contour, -1, 255, 5)
            cv2.drawContours(track_map, [contour], -1, n, -1)
            cv2.drawContours(areas_map, [contour], -1, area, -1)
            areas.append(cv2.contourArea(contour))
            contoursm.append(contour)
    # print('contour length')
    n_contours = len(contoursm)
    print('n_contours')
    print(n_contours)
    areas = np.array(areas)
    lengths = np.sqrt(areas)
    return contours_map, track_map, areas_map, thresh, contours


def object_tracker():
    kernel = 10
    threshv = 1
    
    print('begin dataframe construction...')

    file=   m.NC_PATH+m.FOLDER+'_output.nc'
    ds=xr.open_dataset(file)
    ct = CentroidTracker()
    ds_total=xr.Dataset()
    new_cmap = c.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)

    for i, time in  enumerate(ds['time'].values[:3]):
        ds_clouds= ds[['cloud_top_pressure','pressure_vel']].sel(time=time)
        ds_clouds=ds_clouds.fillna(0)
        values =ds_clouds['cloud_top_pressure'].values
        values = np.squeeze(values)
        id_map=np.zeros_like(values)
        text_map=np.zeros_like(values)
        contours_map, track_map, areas_map, thresh, contours= countourer(values, kernel, threshv)
        objects, contours = ct.update(contours)
        for (objectID, contour) in contours.items():
            cv2.drawContours(id_map, [contour], -1, objectID, -1)

           
        for (objectID, centroid) in objects.items():
            #print(centroid)
            text = str(objectID)
            cv2.putText(text_map, text, (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, 255, 1, False)
            
        ds_clouds['contours_map']=(['lat','lon'], contours_map)
        ds_clouds['areas_map']=(['lat','lon'], areas_map)
        ds_clouds['track_map']=(['lat','lon'], track_map)
        ds_clouds['id_map']=(['lat','lon'], id_map)
        ds_clouds['text_map']=(['lat','lon'], text_map)


        ds_clouds['thresh_map']=(['lat','lon'], thresh)
        
        if ds_total:
            ds_total=xr.concat([ds_total,ds_clouds],'time' )
        else:
            ds_total=ds_clouds
        

        ds_clouds=ds_clouds.sel(lat=slice(10,25), lon=slice(-91,-80))
        ids=ds_clouds.id_map.values
        ids=np.nan_to_num(ids)
        values, counts = np.unique(ids, return_counts=True)
        count_sort_ind = np.argsort(-counts)
        print(values[count_sort_ind])
        print(counts[count_sort_ind])
        #ds_clouds=ds_clouds.where(ds_clouds.id_map==100)
       

        c.map_plotter(ds_clouds, 'cloud_top_pressure_'+str(i),'cloud_top_pressure', units_label='hpa')
        c.map_plotter(ds_clouds, 'contours_id_'+str(i),'id_map',cmap=new_cmap, vmin=0,vmax=1000)
        c.map_plotter(ds_clouds, 'contours_text'+str(i),'text_map')

    return ds_total
        
    
    

    

def main():
    ds_total=object_tracker()
    print(ds_total)
    
if __name__ == "__main__":
    main()
