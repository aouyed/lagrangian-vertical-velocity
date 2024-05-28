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


def threshold_var(values, threshold):
    values[~np.isnan(values)]=1  
    #values[values<=threshold]=1
    #values=np.nan_to_num(values)
    return values



def date_string(prefix, date):
    d0=datetime(date.year, 1,1)
    delta= date-d0
    day_string=str(round(delta.days)+1)
    date_string=prefix+'_'+date.strftime('s%Y')+day_string+date.strftime('%H%M')
    filename=glob.glob('../data/raw/'+date_string+'*')
    assert len(filename)==1, 'ill defined filename'
    return filename[0]


def main():
  start_date=datetime(2023,7,1,18,0)
  end_date=datetime(2023,7,1,23,40)
  datelist=pd.date_range(start_date, end_date, freq='10min')
  dt=timedelta(minutes=10)
  end_date=end_date-dt
  prefix='OR_ABI-L2-ACHTF-M6_G18'
  
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

      frame1=threshold_var(frame1, 241)
      frame2=threshold_var(frame2, 241)
      diff_frame=frame2-frame1
      plt.imshow(diff_frame)
      plt.show()
      plt.close()
      np.save('../data/processed/'+date.strftime('dthresh_%Y%m%d%H%M.npy'),diff_frame)


if __name__=='__main__':
    print('hello')
    main()