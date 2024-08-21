#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:57:26 2022

@author: aouyed
"""

import cv2
import numpy as np
import xarray as xr
import pandas as pd
import calculators as c
import config as config 
from centroidtracker import CentroidTracker
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 
from scipy import stats

R = 6371000
drad = np.deg2rad(1)
dx = R*drad
dy = R*drad
dt=1200
dt_inv=1/dt
SCALE = 1e4
import metpy.calc as mpcalc
from metpy.units import units

def div_calc(u, v, dx=dx, dy=dy):
    u=u * units['m/s']
    v=v * units['m/s']
    div = mpcalc.divergence(
        u, v, dx=dx, dy=dy)
    div = div.magnitude
    div = SCALE*div
    return div


def vort_calc(u, v, dx, dy):
    u=u * units['m/s']
    v=v * units['m/s']
    vort = mpcalc.vorticity(
        u, v, dx=dx, dy=dy)
    vort = vort.magnitude
    vort = SCALE*vort
 
    return  vort


def grad_quants(ds, ulabel, vlabel, dx, dy):
    u = ds[ulabel].values
    v = ds[vlabel].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    div =div_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy())
    vort = vort_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy())

    return div, vort



def grad_calculator(ds):
    lat = ds.lat.values
    lon = ds.lon.values

   
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    div, vort = grad_quants(
        ds, 'u', 'v', dx, dy)
    ds['divergence'] = (['lat', 'lon'], div)
    ds['vorticity'] = (['lat', 'lon'], vort)

    return ds


def calc(ds):
    dx_conv=abs(np.cos(np.deg2rad(ds.lat)))
    ds['u']= dx*dx_conv*dt_inv*ds['flow_x']
    ds['v']= dy*dt_inv*ds['flow_y'] 
    ds_total=xr.Dataset()
    for time in tqdm(ds['time'].values):
        ds_unit=ds.sel(time=time)
        ds_unit= grad_calculator(ds_unit)
        if ds_total:
            ds_total=xr.concat([ds_total, ds_unit], 'time')
        else:
            ds_total=ds_unit
    ds_total.to_netcdf('../data/processed/clouds_mean_kin.nc')
    return ds_total


def main():
    ds_total=xr.open_dataset('../data/processed/clouds_mean.nc')
    #ds_total=ds_total.sel(lat=slice(0,23))
    ds_total=calc(ds_total)
    breakpoint()

    dx_conv=abs(np.cos(np.deg2rad(ds_total.lat)))
    ds_total['u']= dx*dx_conv*dt_inv*ds_total['flow_x']
    ds_total['v']= dy*dt_inv*ds_total['flow_y']
    ds_total= grad_calculator(ds_total)
   
    
    m.plot_loop(ds_total, 'cloud_top_pressure', c.quiver_hybrid, 200, 1000,'viridis',config.FOLDER +'_amv')
    
if __name__ == "__main__":
    main()
