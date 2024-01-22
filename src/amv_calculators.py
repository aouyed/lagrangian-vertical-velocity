#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:48:06 2024

Try to do a quick amv using two consecutive fields and start from there. ]\
    

@author: amirouyed
"""


import xarray as xr
from skimage.registration import optical_flow_tvl1
import numpy as np

def flow_ds(x,y, flowx, flowy, ):
    dims = ('x', 'longitude')

    coords = {
          'latitude': lats,
          'longitude': lons, 
      }
    empty_data = np.nan * np.ones((coords['latitude'].shape[0], coords['longitude'].shape[0])) #,coords['plev'].shape[0]))  # Fill with NaN or other appropriate fill value
    
    
    empty_ds = xr.Dataset(
          {'u_track': (dims, empty_data),'v_track': (dims, empty_data),'u_era5': (dims, empty_data),'v_era5': (dims, empty_data)
           ,'q_era5': (dims, empty_data),'flowx':(dims, empty_data),'flowy':(dims, empty_data)},
          coords=coords)
    return empty_ds



def frame_retreiver(ds, label):
    ds_inpaint=ds
    frame= np.squeeze(ds[label].values)
    frame=np.nan_to_num(frame)
    return frame


def calc(frame0, frame, Lambda):
    if frame0.shape != frame.shape:
        frame=np.resize(frame, frame0.shape)
    
    flowy, flowx=optical_flow_tvl1(frame0, frame, attachment=Lambda)

    return flowx, flowy


def flow_calculator(ds1,ds2):


    frame0=frame_retreiver(ds1,'HT')
    frame=frame_retreiver(ds2, 'HT')
    flowx, flowy=calc(frame0, frame, 10)
    breakpoint()
    #ds_amv=empty_ds(lats,lons)

    # ds_amv['flowx']=(['latitude','longitude'],flowx)
    # ds_amv['flowy']=(['latitude','longitude'],flowy)

    

    
    #return ds_amv


ds1=xr.open_dataset('../data/raw/OR_ABI-L2-ACHAF-M6_G16_s20201890000227_e20201890009535_c20201890011307.nc')
ds2=xr.open_dataset('../data/raw/OR_ABI-L2-ACHAF-M6_G16_s20201890010227_e20201890019535_c20201890021219.nc')
flow_calculator(ds1,ds2)
