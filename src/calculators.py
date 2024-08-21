#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:17:25 2024

@author: amirouyed
"""

import numpy as np
from datetime import datetime 
import pandas as pd
import xarray as xr
from datetime import timedelta
from tqdm import tqdm 
import glob
import matplotlib.pyplot as plt
import cv2
from parameters import parameters

def threshold_var(values, marker):
    values[~np.isnan(values)]=marker  
    #values[values<=threshold]=1
    values=np.nan_to_num(values)
    return values
    
def warp_flow(img, flowx, flowy):
    h, w = flowx.shape
    #flowx[:]=100
    #flowy[:]=0
    flowx = -flowx
    flowy=-flowy
    flowx += np.arange(w)
    flowy += np.arange(h)[:, np.newaxis]
    flowx = flowx.astype(np.float32)
    flowy=flowy.astype(np.float32)
    res = cv2.remap(img, flowx, flowy, cv2.INTER_LINEAR)
    return res


def date_string(prefix, date):
    d0=datetime(date.year, 1,1)
    delta= date-d0
    day_string=str(round(delta.days)+1)
    date_string=prefix+'_'+date.strftime('s%Y')+day_string+date.strftime('%H%M')
    filename=glob.glob('../data/raw/'+date_string+'*')
    assert len(filename)==1, 'ill defined filename'
    return filename[0]


def quick_plot():
    plt.imshow(frame1[frame_slice],vmin=-1,vmax=2)
    plt.colorbar(location='bottom')
    plt.show()
    plt.close()
    plt.imshow(frame2[frame_slice],vmin=-1,vmax=2)
    plt.colorbar(location='bottom')
    plt.show()
    plt.close()
    
    
def threshold_difference(frame1,frame2, warped_frame12, date, param):
    tframe1=threshold_var(frame1.copy(), 1)
    tframe2=threshold_var(frame2.copy(),2)
    warped_tframe1=warping(tframe1, date, param.Lambda)
    warped_tframe12=threshold_var(warped_frame12.copy(), 1)
    df=tframe2-tframe1

    
    tdiff_frame=tframe2-warped_tframe1
    tdiff_frame2=tframe2-warped_tframe12

    overlap_diff=tdiff_frame-df
    overlap_diff2=tdiff_frame2-df

    

  
    param.var_label='flagged_warped_dthresh'
    np.save(param.var_pathname(date),tdiff_frame)
    param.var_label='flagged_dthresh'

    np.save(param.var_pathname(date),df)
    param.var_label='flagged_diff_dthresh'
    np.save(param.var_pathname(date),overlap_diff)
    param.var_label='flagged_diff_dthresh2'
    np.save(param.var_pathname(date),overlap_diff2)




def temp_filter(values,date):
    param_temp=parameters()
    param_temp.prefix='OR_ABI-L2-ACHTF-M6_G18'
    param_temp.var='TEMP'
    filename=param_temp.date_string(date)
    ds=xr.open_dataset(filename)
    values_temp=ds[param_temp.var].values
    values_temp=values_temp[param_temp.frame_slice]
    values[values_temp>param_temp.temp_thresh]=np.nan
    return values     
    
    

def warping(frame,date, Lambda):
    if Lambda=='random':
        flowd=np.random.uniform(low=-1,high=1, size=(frame.shape[0],frame.shape[1],2))
    else:
        flowd=np.load('../data/processed/l'+str(Lambda)+'_'+date.strftime('flagged_amv_%Y%m%d%H%M.npy'))
    
    
    flowx=flowd[:,:,0]
    flowy=flowd[:,:,1]
    frame= warp_flow(frame, flowx, flowy)
    return frame

def main(param):
  
  # abi_lat, abi_lon, deltas=metrics(file_path)
  for date in tqdm(param.calc_datelist()):
      date=date.to_pydatetime()
      date_plus=date+param.dt
  
      file_path1= param.date_string(date)
      file_path2=param.date_string(date_plus)
      ds1=xr.open_dataset(file_path1)
      ds2=xr.open_dataset(file_path2)
      frame1=ds1[param.var].values
      frame2=ds2[param.var].values
      frame1=frame1[param.frame_slice]
      frame2=frame2[param.frame_slice]
      frame1=temp_filter(frame1,date)
      frame2=temp_filter(frame2,date)
      #frame1[frame1>param.temp_thresh]=np.nan
      #frame2[frame2>param.temp_thresh]=np.nan
      
      
      warped_frame1=warping(frame1,date,param.Lambda)
      threshold_difference(frame1.copy(),frame2.copy(),warped_frame1.copy(), date, param)

      warped_df=frame2-warped_frame1
      df=frame2-frame1
      diff_df=warped_df-df
      param.var_label='flagged_warped_d'+ param.var
      np.save(param.var_pathname(date),warped_df)
      param.var_label='flagged_d'+ param.var
      np.save(param.var_pathname(date),df)
      param.var_label='flagged_diff_d'+ param.var
      np.save(param.var_pathname(date),diff_df)



     

if __name__=='__main__':
    param=parameters()
    

    main(param)