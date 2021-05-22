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
        



def main():
    ds_m=xr.open_dataset('../data/raw/reanalysis/omega_T_U_V_01_06_21.nc')
    print(ds_m['w'])
    #ds=interpolation()
    ds=xr.open_dataset('../data/processed/model.nc')
    ds['vel_error']=ds['height_vel']-ds['height_tendency']
    ds['p_error']=ds['pressure_vel']-ds['pressure_tendency']
    ds=ds.where(ds['cloud_top_pressure']>850)
    
    ds['entrainment']=ds['height_vel']-ds['w']
    calc.marginal(ds,'entrainment')
    calc.marginal(ds,'vel_error')
    calc.marginal(ds,'height_vel')
    calc.marginal(ds,'height_tendency')
    calc.marginal(ds,'pressure_vel')
    calc.marginal(ds,'pressure_tendency')
    calc.marginal(ds,'w')
    calc.marginal(ds,'omega')
    # ds['entrainment'].plot.hist(bins=100)
    # plt.show()
    # plt.close()
    # ds['entrainment'].plot.hist(bins=100)
    # plt.show()
    # plt.close()
    # ds['height_vel'].plot.hist(bins=100)
    # plt.show()
    # plt.close()
    # ds['height_tendency'].plot.hist(bins=100)
    # plt.show()
    # plt.close()
    print(abs(ds['vel_error']).mean())
    print(abs(ds['height_vel']).mean())
    print(abs(ds['height_tendency']).mean())
    print(abs(ds['entrainment']).mean())
    print(abs(ds['w']).mean())
    print(abs(ds['pressure_vel']).mean())
    print(abs(ds['pressure_tendency']).mean())
    print(abs(ds['p_error']).mean())
    


if __name__ == '__main__':
    main()
