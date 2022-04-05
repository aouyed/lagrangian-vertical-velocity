
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


class cloudy_system:
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
    def __init__(self,clouds):
        self.clouds=clouds
        self.object_tracker()
        self.dt=clouds.dt
        
        
    
        
    def object_tracker(self):
        self.labels=['cloud_top_pressure','pressure_vel','size_map','pressure_tendency','pressure_adv','vorticity','divergence']

        ds=self.clouds.ds_amv
        try:
            self.ds_raw=xr.open_dataset('../data/processed/tracked_'+str(self.dt)+'.nc')
            self.ds_contours=pickle.load(open( "../data/processed/tracked_contours_"+str(self.dt)+".p", "rb" ))
        except:
            print('calculating contours')
            self.contour_loop(ds)  
            pickle.dump(self.ds_contours,open( "../data/processed/tracked_contours_"+str(self.dt)+".p", "wb" ))
            self.ds_raw=kt.calc(self.ds_raw)
            self.ds_raw=self.ds_raw.where(self.ds_raw.cloud_top_pressure > 0)
            self.ds_raw['size_map']=np.sqrt(self.ds_raw.area_map)
            self.ds_raw['pressure_vel']=100*self.ds_raw['pressure_vel']
            self.ds_raw['pressure_tendency']=100*self.ds_raw['pressure_tendency']
            self.ds_raw['pressure_adv']= self.ds_raw['pressure_vel']- self.ds_raw['pressure_tendency']

            self. ds_raw.to_netcdf('../data/processed/tracked_'+str(self.dt)+'.nc')

        
        self.mean_time()
        
    def contour_loop(self, ds):
        kernel = config.KERNEL
        threshv = config.THRESH
        ct = CentroidTracker()
        ds_total=xr.Dataset()
        contours_dict={'date': [], 'id':[], 'contour':[]}
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
        
        df=pd.DataFrame(data=contours_dict)
        df=df.set_index(['date','id'])
        self.ds_raw=ds_total
        self.ds_contours=xr.Dataset.from_dataframe(df)
          
    
    def mean_time(self):
        try:
            self.ds_time_series=xr.open_dataset('../data/processed/time_series_complete_mean_'+str(self.dt)+'.nc')
        except:
            print('running time series calculator')
            self.time_series_calc()
            self.ds_time_series.to_netcdf('../data/processed/time_series_complete_mean_'+str(self.dt)+'.nc')
      
        try:
            self.ds_clouds_mean=xr.open_dataset('../data/processed/clouds_mean_'+str(self.dt)+'.nc')
        except:
            print('running mean_clouds...')
            labels=self.labels
            labels.append('pressure_rate')
            labels.append('size_rate')

            self.mean_clouds(labels)
            self.ds_clouds_mean.to_netcdf('../data/processed/clouds_mean_'+str(self.dt)+'.nc')
        
    def time_loop(self, data):

        for time in tqdm(self.ds_raw['time'].values):
            #ds=ds_total.sel(time = time)  
            ds_unit=self.ds_raw.sel(time=time)
            ids=ds_unit['id_map'].values
            ids=np.unique(ids[~np.isnan(ids)])
            
            for idno in ids:
                ds=ds_unit.where(ds_unit.id_map==idno)
                data['time'].append(time)
                data['id'].append(idno) 
                for label in self.labels:
                    values=ds[label].mean(skipna=True).item()
                    data[label].append(values)
                 
        return data
        
    def time_series_calc(self):
        ids=self.ds_raw['id_map'].values
        ids=np.unique(ids[~np.isnan(ids)])
        data={}
        for label in self.labels:
            data[label]=[]
        data['time']=[]
        data['id']=[]
        data=self.time_loop(data)
        df=pd.DataFrame(data)
        df=df.set_index(['time','id'])
        self.ds_time_series=xr.Dataset.from_dataframe(df)
        self.ds_time_series['size_rate']=self.ds_time_series['size_map'].differentiate("time",datetime_unit='h' )
        self.ds_time_series['pressure_rate']=100*self.ds_time_series['cloud_top_pressure'].differentiate("time",datetime_unit='s' )
              

    def mean_clouds(self, labels):
        print('means')
        ds_total=xr.Dataset()
        for date in tqdm(self.ds_raw['time'].values):
            ds_unit=self.ds_raw.sel(time=date)
            ids=ds_unit['id_map'].values
            ids=np.unique(ids[~np.isnan(ids)])
            for label in labels:
                img=np.zeros_like(ds_unit['cloud_top_pressure'].values)
                for idno in ids[ids>0]:
                    da=self.ds_contours.sel(date=date, id=idno)
                    contour=da['contour'].values
                    contour=np.squeeze(contour).item()
                    ds_time=self.ds_time_series[label].sel(id=idno, time=date) 
                    try:         
                        stat_value=ds_time.item()
                        cv2.drawContours(img, [contour], -1,stat_value, -1)
                        
                    except Exception as e:
                        print(e)
                        
                    img[img==0]=np.nan
                    ds_unit[label+'_mean']=(['lat','lon'], img)
            
            ds_unit=ds_unit.expand_dims('time')
            if not ds_total:
                ds_total=ds_unit
            else:
                ds_total=xr.concat([ds_total, ds_unit], 'time')
        self.ds_clouds_mean=ds_total
        
    
            
