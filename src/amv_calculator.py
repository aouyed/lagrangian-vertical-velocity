
import cloud_calculators as cc
import main as m 
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
import animator
from natsort import natsorted
import glob

class amv_calculator:
    """An object that tracks and tags clouds, and calculates kinematic 
    and thermodynamic properties `.
    Parameters
    ----------
    file : string, 
        Input file with calculated AMVs.
    
    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated robust location.
    
    """
    def __init__(self):
        self.preprocessing()
        
    def preprocessing(self):
        files=natsorted(glob.glob('../data/interim/'+m.FOLDER+'/*'))
        ds_total = xr.Dataset()
        for file in tqdm(files):
            ds_unit=xr.open_dataset(file)
            if not ds_total:
                ds_total = ds_unit
            else:
                ds_total = xr.concat([ds_total, ds_unit], 'time')
        ds_total=ds_total.sortby('time')
        self.ds_raw=ds_total        
        ds_total.to_netcdf(m.NC_PATH+m.FOLDER+'_raw.nc')
        
    def calc():
        ds=self.ds_raw
        ds=ds.sortby('time')
        times=ds['time'].values[1:-1]
        times_minus=ds['time'].values[0:-2]
        times_plus=ds['time'].values[2:]
        for i,time in enumerate(times):
            ds_unit_past=self.half_amv(ds, times_minus[i], time)
            ds_unit_future=self.half_amv(ds, time, times_plus[i])

    
    def half_amv(ds,time_minus, time):
          ds_unit_minus=ds.sel(times=time_minus, method='nearest')
          ds_unit=ds.sel(time=time, method='nearest')
          frame_minus, height_minus, pressure_minus=self.frame_loader(ds_unit_minus)
          frame, height, pressure=self.frame_loader(ds_unit)
          ds_unit  =self.amv_calc(ds_unit,frame_minus,pressure_minus,height_minus)
          return ds_unit
        
    
    def frame_loader(ds_unit):
        frame=np.squeeze(ds_unit[flow_var].values)
        frame=np.nan_to_num(frame0)
        height=np.squeeze(ds_unit['cloud_top_height'].values)
        pressure=np.squeeze(ds_unit['cloud_top_pressure'].values)
        return frame, height, pressure   
    
    def amv_calc(ds_unit, frame0, pressure0, height0):
        frame=np.squeeze(ds_unit[flow_var].values)
        frame=np.nan_to_num(frame)
        height=np.squeeze(ds_unit['cloud_top_height'].values)
        mask=np.isnan(height)
        pressure=np.squeeze(ds_unit['cloud_top_pressure'].values)
        nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        nframe = cv2.normalize(src=frame, dst=None,
                               alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #need to test this 
        #optical_flow = cv2.optflow.createOptFlow_DeepFlow()
        #flowd = optical_flow.calc(frame0, frame, None)
        flowd=cv2.calcOpticalFlowFarneback(nframe0,nframe, None, 0.5, 3, 20, 3, 7, 1.2, 0)
        # ####
      
        height0=np.nan_to_num(height0)
        height0d=warp_flow(height0.copy(),flowd.copy())
        pressure0d=warp_flow(pressure0.copy(),flowd.copy())
        dz=height-height0d
        pz=height-height0
        dp=pressure-pressure0d
        pp=pressure-pressure0
        dz[mask]=np.nan
        dp[mask]=np.nan
        pp[mask]=np.nan
        pz[mask]=np.nan
        print('mean error:')
        print(np.nanmean(abs(pp)))
            
        dzdt=1000/1200*dz
        dpdt=1/1200*dp
        pppt=1/1200*pp
        pzpt=1000/1200*pz
        
        ds_unit['flow_x']=(('time','lat','lon'),np.expand_dims(flowd[:,:,0],axis=0))
        ds_unit['flow_y']=(('time','lat','lon'),np.expand_dims(flowd[:,:,1],axis=0))
        ds_unit=wind_calculator(ds_unit)
        ds_unit['height_tendency']=(('time','lat','lon'),np.expand_dims(pzpt,axis=0))
        ds_unit['height_vel']=(('time','lat','lon'),np.expand_dims(dzdt,axis=0))
        ds_unit['pressure_vel']=(('time','lat','lon'),np.expand_dims(dpdt,axis=0))
        ds_unit['pressure_tendency']=(('time','lat','lon'),np.expand_dims(pppt,axis=0))
        ds_unit['height0']=(('time','lat','lon'),np.expand_dims(height0,axis=0))
        ds_unit['height0d']=(('time','lat','lon'),np.expand_dims(height0d,axis=0))
        print('frame0')
        map_plotter(ds_unit, 'height0', 'height0')
        print('frame0d')
        map_plotter(ds_unit, 'height0d', 'height0d')
        map_plotter(ds_unit, 'cloud_top_height', 'cloud_top_height')
        map_plotter(ds_unit, 'cloud_top_pressure', 'cloud_top_pressure')
        frame0=frame
        pressure0=pressure
        height0=height
        nframe0=nframe
        print(abs(ds_unit['pressure_vel']).mean())
        print(abs(ds_unit['pressure_tendency']).mean())
        return ds_unit, frame0, height0, pressure0
    
    
    
    
x=amv_calculator()
