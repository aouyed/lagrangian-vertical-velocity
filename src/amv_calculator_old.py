
import cv2
import numpy as np
import xarray as xr
import pandas as pd
import kinematics_tracker as kt
import config as config 
from centroidtracker import CentroidTracker
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 
from scipy import stats
from natsort import natsorted
import glob
flow_var='cloud_top_pressure'
GRID=0.018
R = 6371000
class amv_calculator_old:
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
    def __init__(self, dt):
        self.dt=dt
        self.preprocessing()
        self.calc()
        self.morphing()
        
    def preprocessing(self):
        try:
            self.ds_raw=xr.open_dataset(config.NC_PATH+config.FOLDER+'_raw_'+str(self.dt)+'.nc')
        except:
            files=natsorted(glob.glob('../data/interim/'+config.FOLDER+'/*'))
            ds_total = xr.Dataset()
            for file in tqdm(files):
                ds_unit=xr.open_dataset(file)
                if not ds_total:
                    ds_total = ds_unit
                else:
                    ds_total = xr.concat([ds_total, ds_unit], 'time')
            ds_total=ds_total.sortby('time')
            self.ds_raw=ds_total        
            ds_total.to_netcdf(config.NC_PATH+config.FOLDER+'_raw_'+str(self.dt)+'.nc')
        
    def calc(self):
        try:
            self.ds_amv=xr.open_dataset(config.NC_PATH+config.FOLDER+'_amv_'+str(self.dt)+'.nc')
        except:
            ds=self.ds_raw
            ds=ds.sortby('time')
            timedelta=np.timedelta64(self.dt, 's')
            times=pd.date_range(start=ds['time'].values[0], end=ds['time'].values[-1], freq=str(self.dt)+'s')
            ds_total = xr.Dataset()
            for i in tqdm(range(len(times[:-1]))):
                ds_unit=self.prepare_amv(ds, times[i], times[i+1])
                if not ds_total:
                    ds_total = ds_unit
                else:
                    ds_total = xr.concat([ds_total, ds_unit], 'time')
            ds_total=ds_total.sortby('time')
            self.ds_amv=ds_total        
            ds_total.to_netcdf(config.NC_PATH+config.FOLDER+'_amv_'+str(self.dt)+'.nc')

    
    def prepare_amv(self, ds,time_minus, time):
          ds_unit_minus=ds.sel(time=time_minus, method='nearest')
          ds_unit=ds.sel(time=time, method='nearest')
          frame_minus, height_minus, pressure_minus=self.frame_loader(ds_unit_minus)
          frame, height, pressure=self.frame_loader(ds_unit)
          ds_unit  =self.amv_calc(ds_unit,frame_minus,pressure_minus,height_minus)
          return ds_unit
        
    
    def frame_loader(self, ds_unit):
        frame=np.squeeze(ds_unit[flow_var].values)
        frame=np.nan_to_num(frame)
        height=np.squeeze(ds_unit['cloud_top_height'].values)
        pressure=np.squeeze(ds_unit['cloud_top_pressure'].values)
        return frame, height, pressure   
    
    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        flow = flow.astype(np.float32)
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res
    def wind_calculator(self, ds):
        flow_x=np.squeeze(ds['flow_x'].values)
        flow_y=np.squeeze(ds['flow_y'].values)
        lon,lat=np.meshgrid(ds['lon'].values,ds['lat'].values)
        dthetax = GRID*flow_x
        dradsx = dthetax * np.pi / 180
        lat = lat*np.pi/180
        dx = R*abs(np.cos(lat))*dradsx
        u= dx/1800
        
        dthetay =GRID*flow_y
        dradsy = dthetay * np.pi / 180
        dy = R*dradsy
        v= dy/1800
        
        ds['u']=(('lat','lon'),u)
        ds['v']=(('lat','lon'),v)
        return ds
    
    
    def morphing(self):
        print('morphing')
        labels=['cloud_top_height','cloud_top_pressure','flow_x','flow_y']
        times=self.ds_amv['time'].values
        ds_total=xr.Dataset()
        for i in tqdm(range(len(times[:-2]))):
            ds_unit_t0=self.ds_amv[labels].sel(time=times[i])
            ds_unit_t1=self.ds_amv[labels].sel(time=times[i+1])
            ds_unit_t2=self.ds_amv[labels].sel(time=times[i+2])
            pressure_t0=ds_unit_t0['cloud_top_pressure'].values
            pressure_t1=ds_unit_t1['cloud_top_pressure'].values
            pressure_t2=ds_unit_t2['cloud_top_pressure'].values
            flow_shape=ds_unit_t0['flow_x'].values.shape
            flow_shape=(flow_shape[0],flow_shape[1], 2)
            flowd_tl=np.zeros(flow_shape)
            flowd_tu=np.zeros(flow_shape)

            flowd_tl[:,:,0]=ds_unit_t0['flow_x'].values
            flowd_tl[:,:,1]=ds_unit_t0['flow_y'].values
            ds_unit_h1= self.half_warper(pressure_t0, pressure_t1, flowd_tl, ds_unit_t1.copy())
                  
            flowd_tu[:,:,0]=-ds_unit_t2['flow_x'].values
            flowd_tu[:,:,1]=-ds_unit_t2['flow_y'].values
            ds_unit_h2= -self.half_warper(pressure_t2, pressure_t1, flowd_tu, ds_unit_t1.copy())
            ds_unit= 0.5*(ds_unit_h1+ds_unit_h2) 
            if  ds_total:
                ds_total=xr.concat([ds_total, ds_unit],'time')
            else:
                ds_total=ds_unit
        self.ds_amv['dp_morph']=ds_total['dp_morph']
        self.ds_amv['pp_morph']=ds_total['pp_morph']        
                      
                         
                    
                  
                    
    def half_warper(self, pressure0, pressure, flowd, ds_unit):
            mask=np.isnan(pressure)
            pressure0d=self.warp_flow(pressure0.copy(),flowd.copy())
            dp=pressure-pressure0d
            pp=pressure-pressure0
            dp[mask]=np.nan
            pp[mask]=np.nan
            ds_unit['dp_morph']=(('lat','lon'),dp)
            ds_unit['pp_morph']=(('lat','lon'),pp)

            return ds_unit[['dp_morph','pp_morph']]
            
                
            
            return dz, pz, dp, pp

    def amv_calc(self, ds_unit, frame0, pressure0, height0):
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
        height0d=self.warp_flow(height0.copy(),flowd.copy())
        pressure0d=self.warp_flow(pressure0.copy(),flowd.copy())
        dz=height-height0d
        pz=height-height0
        dp=pressure-pressure0d
        pp=pressure-pressure0
        dz[mask]=np.nan
        dp[mask]=np.nan
        pp[mask]=np.nan
        pz[mask]=np.nan
            
        dzdt=dz
        dpdt=dp
        pppt=pp
        pzpt=pz
        
        ds_unit['flow_x']=(('lat','lon'),flowd[:,:,0])
        ds_unit['flow_y']=(('lat','lon'),flowd[:,:,1])
        ds_unit=self.wind_calculator(ds_unit)
        ds_unit['pressure_vel']=(('lat','lon'),dp)
        ds_unit['pressure_tendency']=(('lat','lon'),pp)
        frame0=frame
        pressure0=pressure
        height0=height
        nframe0=nframe
        return ds_unit
    