
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
        self.ds_clouds_mean_rolled=None
        self.ds_clouds_rolled=None
        self.ds_time_series_rolled=None
        self.ds_raw=None
        self.ds_contour=None
        
        self.clouds=clouds
        self.dt=clouds.dt
        self.object_tracker()
        self.rolling_mean()
        
        
    
        
    def object_tracker(self):
        self.labels=['cloud_top_pressure','cloud_top_temperature','pressure_vel','size_map','pressure_tendency','pressure_adv','vorticity','divergence','dp_morph','pp_morph']

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
            ds_clouds= ds[['cloud_top_pressure','pressure_vel','pressure_tendency','flow_x','flow_y','cloud_top_temperature','dp_morph','pp_morph']].sel(time=time)
            ds_clouds['cloud_top_temperature']=ds_clouds['cloud_top_temperature'].where(ds_clouds.cloud_top_temperature < config.temp_thresh)
            ds_clouds=ds_clouds.fillna(0)
            values =ds_clouds['cloud_top_temperature'].values
            #values = values[values<config.temp_thresh]
            values = np.squeeze(values)    
            contours= self.contourer(values, kernel, threshv)
            objects, contours = ct.update(contours)
            ds_clouds= self.contour_drawer(contours, objects,ds_clouds)
            #contour_plotter(ds_clouds, new_cmap, i)
            self.contour_store(contours_dict, time, contours )
            
            
            if ds_total:
                ds_total=xr.concat([ds_total,ds_clouds],'time' )
            else:
                ds_total=ds_clouds
        
        df=pd.DataFrame(data=contours_dict)
        df=df.set_index(['date','id'])
        self.ds_raw=ds_total
        self.ds_contours=xr.Dataset.from_dataframe(df)
        
    def contourer(self, values, kernelv, threshv,  area_lim=100):
        values = cv2.blur(values, (10, 10))
    
        _, thresh = cv2.threshold(values.copy(), threshv, 255,cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)
        kernel = np.ones((kernelv, kernelv), np.uint8)
    
        contours, hierarchy = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours=self.contour_filter(contours)
        return contours
          
    def contour_drawer(self, contours, objects, ds_clouds):
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
    
    
    def contour_filter(self, contours):
        dummy_c=[]
        for contour in contours:
            M=cv2.moments(contour)
            if M['m00'] >0:
                dummy_c.append(contour)
                
        contours=dummy_c
        return contours
    
    def contour_store(self, contours_dict,date, contours ):
        for (objectID, contour) in contours.items():
            contours_dict['id'].append(objectID)
            contours_dict['contour'].append(contour)
            contours_dict['date'].append(date)
    
    
    
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
            self.rolling_mean()
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
        
    def rolling_mean(self):
        self.ds_clouds_mean_rolled=self.ds_clouds_mean.rolling(time=3, center=True).mean()
        self.ds_clouds_rolled=self.ds_raw.rolling(time=3, center=True).mean()
        self.ds_time_series_rolled=self.ds_time_series.rolling(time=3, center=True).mean()

    
            
