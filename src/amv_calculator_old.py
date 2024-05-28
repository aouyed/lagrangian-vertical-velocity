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
    
    flowy, flowx=optical_flow_tvl1(frame0, frame, attachment=Lambda)

    return flowx, flowy


def flow_calculator(ds1,ds2, file_path1, file_path2):
        
    ds1=xr.open_dataset(file_path1)
    ds2=xr.open_dataset(file_path2)
    file_id = Dataset(file_path1)
    ds1['HT'].plot()
    plt.show()
    plt.close()
    
    abi_lat, abi_lon = calculate_degrees(file_id)
    abi_lat=abi_lat.filled(np.nan)
    abi_lon=abi_lon.filled(np.nan)
    deltas=lat_lon_grid_deltas(abi_lon, abi_lat)
    dx=deltas[-1].magnitude
    dy=deltas[-2].magnitude
    

    
    frame0, mask0=frame_retreiver(ds1,'HT')
    frame, mask=frame_retreiver(ds2, 'HT')
    
    flowx, flowy=calc(frame0, frame, 1)
    #frame_warped=warp_flow(frame0.astype(np.float32), flowx, flowy)
    #breakpoint()
    flowx[mask0]=np.nan
    flowx[mask]=np.nan
    flowy[mask0]=np.nan
    flowy[mask]=np.nan
    dx=dx.copy()
    dy=dy.copy()
    dx.resize( flowx.shape )
    dy.resize(flowy.shape )

    ds_amv=flow_ds(ds1['x'].values,ds1['y'].values, flowx, flowy)
    ds_amv['u_track']=ds_amv['flowx']*dx/DT
    ds_amv['v_track']=ds_amv['flowy']*dy/DT
    ds_amv['dx']=(['latitude','longitude'], dx)
    ds_amv['dy']=(['latitude','longitude'], dy)

    
    return ds_amv
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


def warp_ds(ds1, ds_amv):
    frame0, mask0=frame_retreiver(ds1,'HT')
    flowx=ds_amv['flowx'].values
    flowy=ds_amv['flowy'].values
    flowx=np.nan_to_num(flowx)
    flowy=np.nan_to_num(flowy)

    frame_warped=warp_flow(frame0.astype(np.float32), flowx, flowy)

file_path1='../data/raw/OR_ABI-L2-ACHAF-M6_G16_s20201890000227_e20201890009535_c20201890011307.nc'
file_path2='../data/raw/OR_ABI-L2-ACHAF-M6_G16_s20201890010227_e20201890019535_c20201890021219.nc'

# # Print latitude array
ds1=xr.open_dataset(file_path1)
ds_amv=flow_calculator(file_path1,file_path2)
ds_amv['speed']=np.sqrt(ds_amv.u_track**2+ds_amv.v_track**2)
ds_amv=ds_amv.where(ds_amv.speed<100)


# quiver_cartopy(ds_amv, 'u_track', 'v_track')
# warp_ds(ds1, ds_amv)
    