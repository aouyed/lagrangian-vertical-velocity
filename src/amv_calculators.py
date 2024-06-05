#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:48:06 2024

Try to do a quick amv using two consecutive fields and start from there. ]\
Try to do some quick warping and then see how smooth the profiles are

create the ds_amv, then you can filter it and feed it to the warping algorithm again by extracting the flow variables and inpainting them w zeroes 
    

@author: amirouyed
"""


import xarray as xr
from skimage.registration import optical_flow_tvl1
import numpy as np
import matplotlib.pyplot as plt
from metpy.calc import lat_lon_grid_deltas
from metpy.calc import advection 

from netCDF4 import Dataset
from scipy import ndimage as nd
import cv2
from datetime import datetime
from datetime import timedelta 
from tqdm import tqdm 
import pandas as pd
import glob 
import cv2

DT=10*60

def fill(data, invalid=None):
    """
    (algorithm created by Juh_:https://stackoverflow.com/users/1206998/juh)
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. 
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """    
    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, 
                                    return_distances=False, 
                                    return_indices=True)
    return data[tuple(ind)]

def calculate_degrees(file_id):
    
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon

def quiver_cartopy(ds, u, v):
    fig, ax = plt.subplots()

    ds=ds.coarsen(x=20, y=20, boundary='trim').median()
    X, Y = np.meshgrid(ds['x'].values, ds['y'].values)
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values),scale=500)
    qk=ax.quiverkey(Q, 0.5, 0.5, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    #ax.coastlines()
    #gl=None
    #gl=ax.gridlines(draw_labels=False, x_inline=False, y_inline=False)
    plt.show()
    plt.close()
    #gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])


def flow_ds(x,y, flowx, flowy):
    dims = ('x', 'y')

    coords = {
          'x': x,
          'y': y, 
      }
    
    amv_ds = xr.Dataset(
          {'flowx': (dims, flowx)
           ,'flowy': (dims, flowy)},
          coords=coords)
    return amv_ds



def frame_retreiver(ds, label):
    frame= np.squeeze(ds[label].values)
    mask=np.isnan(frame)
    #frame=fill(frame)
    frame=np.nan_to_num(frame)
    return frame, mask


def calc(frame0, frame, Lambda):
    if frame0.shape != frame.shape:
        frame=frame.resize(frame0.shape)
    nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    nframe = cv2.normalize(src=frame, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    optical_flow=cv2.optflow.DualTVL1OpticalFlow_create()
    optical_flow.setLambda(Lambda)
    flowd = optical_flow.calc(nframe0, nframe, None)

    flowx=flowd[:,:,0]
    flowy=flowd[:,:,1]
     
    return flowd



def metrics(file_path):
    file_id = Dataset(file_path)
    abi_lat, abi_lon = calculate_degrees(file_id)
    abi_lat=abi_lat.filled(np.nan)
    abi_lon=abi_lon.filled(np.nan)
    deltas=lat_lon_grid_deltas(abi_lon, abi_lat)
    dx=deltas[-1].magnitude
    dy=deltas[-2].magnitude
    return abi_lat, abi_lon, deltas



def flow_calculator(file_path1, file_path2,var,Lambda, frame_slice):
        
    ds1=xr.open_dataset(file_path1)
    ds2=xr.open_dataset(file_path2)
    file_id = Dataset(file_path1)
    #ds1[var].plot()
    #plt.show()
    #plt.close()
    
    

    
    frame0, mask0=frame_retreiver(ds1,var)
    frame, mask=frame_retreiver(ds2, var)
    frame0=frame0[frame_slice]
    frame=frame[frame_slice]
    if var=='TEMP':
        frame[frame>250]=0
    flowd=calc(frame0, frame, Lambda)
   

    
    return flowd
    #ds_amv=empty_ds(lats,lons)

    # ds_amv['flowx']=(['latitude','longitude'],flowx)
    # ds_amv['flowy']=(['latitude','longitude'],flowy)

    
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
    

def warp_ds(ds1, ds_amv):
    frame0, mask0=frame_retreiver(ds1,'HT')
    flowx=ds_amv['flowx'].values
    flowy=ds_amv['flowy'].values
    flowx=np.nan_to_num(flowx)
    flowy=np.nan_to_num(flowy)

    frame_warped=warp_flow(frame0.astype(np.float32), flowx, flowy)
    

def main():
    
    start_date=datetime(2023,7,1,18,0)
    end_date=datetime(2023,7,1,23,40)
    datelist=pd.date_range(start_date, end_date, freq='10min')
    dt=timedelta(minutes=10)
    end_date=end_date-dt
    prefix='OR_ABI-L2-ACHTF-M6_G18'
    Lambda=0.07
    frame_slice=np.index_exp[1700:1900, 1500:2500]
    file_path= date_string(prefix, start_date)
    
    # abi_lat, abi_lon, deltas=metrics(file_path)
    for date in tqdm(param.calc_datelist()):
        date=date.to_pydatetime()
        date_plus=date+param.dt
    
        file_path1= param.date_string(date)
        file_path2=param.date_string(date_plus)
        #var='TEMP'
        flowd=flow_calculator(file_path1,file_path2, var, Lambda, frame_slice)
        
        np.save('../data/processed/'+'l'+str(Lambda)+'_'+date.strftime('flagged_amv_%Y%m%d%H%M.npy'),flowd)
     
    
if __name__=='__main__':
    print('hello')
    main()