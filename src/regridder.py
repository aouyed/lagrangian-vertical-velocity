#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:31:22 2021

@author: aouyed
"""
import xarray as xr
import numpy as np
import xesmf as xe
from datetime import datetime
from natsort import natsorted
import glob
from dateutil import parser

FOLDER='05_30_21'

files=natsorted(glob.glob('../data/raw/'+FOLDER+'/G16V04.0.ACTIV*'))
for file in files:
    ds=xr.open_dataset(file)
    ds =ds.rename({'latitude':'lat', 'longitude':'lon'})
    print(ds)
    latmax = ds['lat'].max(skipna=True).item()
    latmin = ds['lat'].min(skipna=True).item()
    lonmax = ds['lon'].max(skipna=True).item()
    lonmin = ds['lon'].min(skipna=True).item()
    #new_lat = np.arange(latmin, latmax, 0.018)
    #new_lon = np.arange(lonmin, lonmax, 0.018)
    new_lat = np.arange(latmin, latmax, 0.031)
    new_lon = np.arange(lonmin, lonmax, 0.031)       
    ds_out = xr.Dataset({'lat': (['lat'], new_lat), 'lon': ('lon', new_lon), })
    regridder = xe.Regridder(ds, ds_out, 'bilinear',reuse_weights=True)
    dr_out = regridder(ds[['temperature_ir','cloud_top_height','cloud_top_temperature','belwp','cloud_top_pressure']])
    date_s=ds.processed_date
    date_s=date_s.replace("00,", "01,")
    date=np.array([parser.parse(date_s).replace(tzinfo=None)])
    dr_out = dr_out.expand_dims('time')
    dr_out = dr_out.assign_coords(time=date)
    filename=date.item().strftime('%Y-%m-%d-%H-%M')+'.nc'
    print(filename)
    dr_out.to_netcdf('../data/interim/'+FOLDER+'/'+ filename)
    #print(dr_out)