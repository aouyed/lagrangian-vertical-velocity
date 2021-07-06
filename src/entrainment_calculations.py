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
import main as m 
import datetime
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
    #ds['entrainment']=ds['pressure_vel']-ds['w']
    #ds['w']=100*ds['w']
    #ds['w*']=100*ds['w*']
    #ds['w_s']=100*ds['w_s']
    ds['vel_error']=ds['vel_error']
    ds['pressure_vel']=100*ds['pressure_vel']
    ds['pressure_tendency']=100*ds['pressure_tendency']
    #ds['entrainment']=100*ds['entrainment']
    ds['p_error']=ds['p_error']

    
    
    
    return ds

def plot_sequence(ds,tag):

    ds=ds.reindex(lat=list(reversed(ds.lat)))
    ds=ds.coarsen(time=24,boundary='trim').mean()
    ds=ds.sel(time=ds['time'].values[0])
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()


    #calc.map_plotter(ds, 'w*_'+tag,'w*', units_label='cm/s', vmin=-1, vmax=1)
    #calc.map_plotter(ds, 'w_s_'+tag,'w_s', units_label='cm/s', vmin=-1, vmax=1)
    #calc.map_plotter(ds, 'w_'+tag,'w', units_label='cm/s', vmin=-1, vmax=1)
    #calc.map_plotter(ds, 'entrainment_'+tag,'entrainment', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'pressure_vel_'+tag,'pressure_vel', units_label='cm/s', vmin=-0.4, vmax=0.4)
    calc.map_plotter(ds, 'cloud_top_pressure_'+tag,'cloud_top_pressure', units_label='hpa')
    calc.marginals(ds, tag)
    calc.post_process(ds,tag)
   
    

def analysis(ds,tag):
    ds=preprocess(ds)
    plot_sequence(ds.where(ds['cloud_top_pressure']<750),tag+'_high')
    plot_sequence(ds.where(ds['cloud_top_pressure']>850),tag+'_low')
    

    #ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
    #calc.quiver_plot(ds.sel(time=ds['time'].values[5]), tag)

    print(ds)

    
def main():

    #ds=xr.open_dataset('../data/processed/model_january.nc')
    
    ds=xr.open_dataset(m.NC_PATH+m.FOLDER+'_output.nc')
    ds=ds[['cloud_top_pressure','pressure_vel']]
   # ds['pressure_vel']=ds['pressure_vel'].fillna(-9999)
    #ds['cloud_top_pressure']=ds['cloud_top_pressure'].fillna(9999)
    #ds=ds.sel(lon=slice(-100,-70),lat=slice(0,30))
    #ds=ds.where(ds['cloud_top_pressure']<850)
    #ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
    #ds=ds.where(ds['cloud_top_pressure']>750)
    ds=ds.coarsen(time=3,boundary='trim').mean()
    ds=ds.coarsen(lat=10, boundary='trim').mean().coarsen(lon=10, boundary='trim').mean()
    ds['pressure_vel']=100*ds['pressure_vel']
    #
    #calc.map_plotter_masked(ds.sel(time=ds['time'].values[0]), 'cloud_top_pressure', 'cloud_top_pressure')
    #m.plot_loop(ds, 'cloud_top_pressure',calc.implot_masked, 200, 1000,'winter',m.FOLDER)
    m.plot_loop(ds, 'pressure_vel', calc.implot_masked, -1, 1,'RdBu',m.FOLDER)
    #analysis(ds, m.FOLDER)
    #ds=xr.open_dataset('../data/processed/model_may.nc')
    #analysis(ds, 'may')
    
    
if __name__ == '__main__':
    main()