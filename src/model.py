#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:53:57 2021

@author: aouyed
"""

import xarray as xr
import calculators as calc
import matplotlib.pyplot as plt
import entrainment_calculations as ec




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
        ds_inter=ec.advection(ds_inter)
        if ds_total:
            ds_total=xr.concat([ds_total,ds_inter], 'time') 
        else:
            ds_total=ds_inter
     
    print(ds_total)
    ds_total.to_netcdf('../data/processed/model_january.nc')
    return ds_total
        
    

def main():
    ds_m=xr.open_dataset('../data/raw/reanalysis/omega_T_U_V_01_06_21.nc')
    #print(ds_m['w'])
    ds=interpolation()
    


if __name__ == '__main__':
    main()
