#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:55:28 2021

@author: aouyed
"""
import xarray as xr
import model
import calculators as calc
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np

def advection(ds):
    dx, dy = mpcalc.lat_lon_grid_deltas(ds['lon'].values, ds['lat'].values)
    gradz=mpcalc.gradient(np.squeeze(ds['cloud_top_height'].values), deltas=(dy, dx))
    gradz_y=gradz[0].magnitude
    gradz_x=gradz[1].magnitude
    
    ds['gradz_y']=(('time','lat','lon'),np.expand_dims(gradz_y,axis=0))
    ds['gradz_x']=(('time','lat','lon'),np.expand_dims(gradz_x,axis=0))
    
    ds['adv']=ds['u']*ds['gradz_x'] + ds['v']*ds['gradz_y']
    ds['adv']=100*ds['adv']
    return ds

def preprocess(ds):
    ds['vel_error']=ds['height_vel']-ds['height_tendency']
    ds['p_error']=ds['pressure_vel']-ds['pressure_tendency']
   
    

    #ds['w']=ds['w*']
    ds['entrainment']=ds['pressure_vel']-ds['w']
    ds['w']=100*ds['w']
    ds['w*']=100*ds['w*']
    ds['w_s']=100*ds['w_s']
    ds['vel_error']=100*ds['vel_error']
    ds['pressure_vel']=100*ds['pressure_vel']
    ds['pressure_tendency']=100*ds['pressure_tendency']
    ds['entrainment']=100*ds['entrainment']
    ds['p_error']=100*ds['p_error']

    
    
    
    return ds

def plot_sequence(ds,tag):
    ds=ds.coarsen(time=24,boundary='trim').mean()
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
    ds=ds.reindex(lat=list(reversed(ds.lat)))

    calc.map_plotter(ds, 'w*_'+tag,'w*', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'w_s_'+tag,'w_s', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'w_'+tag,'w', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'entrainment_'+tag,'entrainment', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'pressure_vel_'+tag,'pressure_vel', units_label='cm/s', vmin=-0.4, vmax=0.4)
    calc.map_plotter(ds, 'cloud_top_pressure_'+tag,'cloud_top_pressure', units_label='hpa')
    calc.marginals(ds, tag)
    calc.post_process(ds,tag)
   
    

def analysis(ds,tag):
    ds=preprocess(ds)
    plot_sequence(ds.where(ds['cloud_top_pressure']<850),tag+'_high')
    plot_sequence(ds.where(ds['cloud_top_pressure']>850),tag+'_low')

    print(ds)

    
def main():
    
    #ds=xr.open_dataset('../data/processed/model.nc')
    #analysis(ds, 'january')
    ds=xr.open_dataset('../data/processed/model_january.nc')
    analysis(ds, 'january')
    
    
if __name__ == '__main__':
    main()