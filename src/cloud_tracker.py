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


def countourer(values, kernelv, threshv,  area_lim=100):
    values = cv2.blur(values, (10, 10))

    _, thresh = cv2.threshold(values.copy(), threshv, 255,cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    kernel = np.ones((kernelv, kernelv), np.uint8)
    #thresh=cv2.blur(thresh,(8,8))
    #thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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
    kernel = 1

    threshv = 1
    print('begin dataframe construction...')

    file=   m.NC_PATH+m.FOLDER+'_output.nc'
    ds=xr.open_dataset(file)
    ct = CentroidTracker()
    
    for time in  ds['time'].values[:2]:
        ds_clouds= ds[['cloud_top_pressure','pressure_vel']].sel(time=time)
        ds_clouds=ds_clouds.fillna(0)
        values =ds_clouds['cloud_top_pressure'].values
        values = np.squeeze(values)
        frame=np.zeros_like(values)
        contours_map, track_map, areas_map, thresh, contours= countourer(values, kernel, threshv)
        objects = ct.update(contours)
        for (objectID, centroid) in objects.items():
            #print(centroid)
            cv2.circle(frame, (centroid[0], centroid[1]), 5,  255, -1)
            text = str(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2, False)
        #frame = frame[::-1,:] 
    
        cv2.imwrite("square_circle_opencv.jpg",frame)
        ds_clouds['contours_map']=(['lat','lon'], contours_map)
        ds_clouds['areas_map']=(['lat','lon'], areas_map)
        ds_clouds['track_map']=(['lat','lon'], track_map)
        ds_clouds['thresh_map']=(['lat','lon'], thresh)
        ds_clouds['centroids']=(['lat','lon'], frame)
        
        ds_clouds=ds_clouds.sel(lat=slice(20,25), lon=slice(-91,-80))
        
        c.map_plotter(ds_clouds, 'cloud_top_pressure_','cloud_top_pressure', units_label='hpa')
        c.map_plotter(ds_clouds, 'contours_','track_map')
        c.map_plotter(ds_clouds, 'contours_','centroids')
    
        #c.map_plotter(ds_clouds, 'contours_','thresh_map')
    

    

def main():
    object_tracker()
    
if __name__ == "__main__":
    main()
