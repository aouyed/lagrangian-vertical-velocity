#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:53:57 2021

@author: aouyed
"""

import xarray as xr
import calculators as calc
import matplotlib.pyplot as plt




def interpolation():
    ds_m=xr.open_dataset('../data/raw/reanalysis/omega_T_U_V_01_06_21.nc')
    ds_s=xr.open_dataset('../data/processed/netcdf/january_output.nc')
    
    ds_m = ds_m.assign_coords(longitude=(((ds_m.longitude + 180) % 360) - 180))
    ds_m=ds_m.sortby('latitude').sortby('longitude')
    print(ds_m)
    
    latmin=ds_s['lat'].min().values.item(0)
    latmax=ds_s['lat'].max().values.item(0)
    
    lonmin=ds_s['lon'].min().values.item(0)
    lonmax=ds_s['lon'].max().values.item(0)
    ds_m=ds_m.sel(latitude=slice(latmin,latmax))
    ds_m=ds_m.sel(longitude=slice(lonmin,lonmax))
    print(ds_m)
    ds_total=xr.Dataset()
    for time in ds_m['time'].values:
        print(time)
        ds_m_unit=ds_m.sel(time=time, method='nearest')
        ds_s_unit=ds_s.sel(time=time, method='nearest')
        ds_inter=calc.interpolation(ds_s_unit,ds_m_unit)
        ds_inter=calc.omega_calculator(ds_inter,'pressure_vel')
        ds_inter=calc.omega_calculator(ds_inter,'pressure_tendency')
        if ds_total:
            ds_total=xr.concat([ds_total,ds_inter], 'time') 
        else:
            ds_total=ds_inter
     
    print(ds_total)
    ds_total.to_netcdf('../data/processed/model.nc')
    return ds_total
        

def post_process(ds):
    calc.marginal(ds,'entrainment')
    calc.marginal(ds,'vel_error')
    calc.marginal(ds,'height_vel')
    calc.marginal(ds,'height_tendency')
    calc.marginal(ds,'pressure_vel')
    calc.marginal(ds,'pressure_tendency')
    calc.marginal(ds,'w')
    calc.marginal(ds,'omega')
    print(abs(ds['vel_error']).mean())
    print(abs(ds['height_vel']).mean())
    print(abs(ds['height_tendency']).mean())
    print(abs(ds['entrainment']).mean())
    print(abs(ds['w']).mean())
    print(abs(ds['w*']).mean())
    print(abs(ds['w_s']).mean())
    print(abs(ds['pressure_vel']).mean())
    print(abs(ds['pressure_tendency']).mean())
    print(abs(ds['p_error']).mean())
    

def main():
    #ds_m=xr.open_dataset('../data/raw/reanalysis/omega_T_U_V_01_06_21.nc')
    #print(ds_m['w'])
    #ds=interpolation()
    ds=xr.open_dataset('../data/processed/model.nc')
    ds['vel_error']=ds['height_vel']-ds['height_tendency']
    ds['p_error']=ds['pressure_vel']-ds['pressure_tendency']
    ds=ds.where(ds['cloud_top_pressure']<700)
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
   

    #ds['w']=ds['w*']
    ds['entrainment']=ds['pressure_vel']-ds['w']
    ds['w']=100*ds['w']
    ds['w*']=100*ds['w*']
    ds['w_s']=100*ds['w_s']
    ds['vel_error']=100*ds['vel_error']
    ds['pressure_vel']=100*ds['pressure_vel']
    ds['pressure_tendency']=100*ds['pressure_tendency']
    ds['entrainment']=100*ds['entrainment']
    post_process(ds)
    
    ds=ds.sel(time=ds['time'].values[12])
    print(ds)
    calc.map_plotter(ds, 'w*','w*', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'w_s','w_s', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'w','w', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'entrainment','entrainment', units_label='cm/s', vmin=-1, vmax=1)
    calc.map_plotter(ds, 'pressure_vel','pressure_vel', units_label='cm/s', vmin=-0.5, vmax=0.5)
    calc.map_plotter(ds, 'cloud_top_pressure','cloud_top_pressure', units_label='hpa')
    calc.quiver_plot(ds, 'model')


if __name__ == '__main__':
    main()
