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
    
    
def threshold_difference(frame1,frame2, date, Lambda):
    tframe1=threshold_var(frame1, 1)
    tframe2=threshold_var(frame2,2)
    warped_tframe1=warping(tframe1, date, Lambda)
   
    df=frame2-frame1

    
    tdiff_frame=tframe2-warped_tframe1

    np.save('../data/processed/l'+str(Lambda)+'_'+date.strftime('flagged_warped_dthresh_%Y%m%d%H%M.npy'),tdiff_frame)
    np.save('../data/processed/l'+str(Lambda)+'_'+date.strftime('flagged_dthresh_%Y%m%d%H%M.npy'),df)




    
    
    

def warping(frame,date, Lambda):
    if Lambda=='random':
        flowd=np.random.uniform(low=-1,high=1, size=(frame.shape[0],frame.shape[1],2))
    else:
        flowd=np.load('../data/processed/l'+str(Lambda)+'_'+date.strftime('flagged_amv_%Y%m%d%H%M.npy'))
    
    
    flowx=flowd[:,:,0]
    flowy=flowd[:,:,1]
    frame= warp_flow(frame, flowx, flowy)
    return frame

def main():
  start_date=datetime(2023,7,1,18,0)
  end_date=datetime(2023,7,1,23,40)
  datelist=pd.date_range(start_date, end_date, freq='10min')
  dt=timedelta(minutes=10)
  end_date=end_date-dt
  prefix='OR_ABI-L2-ACHTF-M6_G18'
  Lambda=0.07
  frame_slice=np.index_exp[1700:1900, 1500:2500]
  param=parameters()


  
  # abi_lat, abi_lon, deltas=metrics(file_path)
  for date in tqdm(datelist):
      date=date.to_pydatetime()
      date_plus=date+dt
  
      file_path1= date_string(prefix, date)
      file_path2=date_string(prefix, date_plus)
      var='TEMP'
      ds1=xr.open_dataset(file_path1)
      ds2=xr.open_dataset(file_path2)
      frame1=ds1[var].values
      frame2=ds2[var].values
      frame1=frame1[frame_slice]
      frame2=frame2[frame_slice]
      frame1[frame1>param.temp_thresh]=np.nan
      frame2[frame2>param.temp_thresh]=np.nan

      threshold_difference(frame1,frame2,date, Lambda)
      
      warped_frame1=warping(frame1,date,Lambda)
      warped_df=frame2-warped_frame1
      df=frame2-frame1
      np.save('../data/processed/l'+str(Lambda)+'_'+date.strftime('flagged_warped_d'+var+'_%Y%m%d%H%M.npy'),warped_df)
      np.save('../data/processed/l'+str(Lambda)+'_'+date.strftime('flagged_d'+var+'_%Y%m%d%H%M.npy'),df)


     

if __name__=='__main__':
    print('hello')
    main()