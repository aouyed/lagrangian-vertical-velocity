#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:53:57 2021

@author: aouyed
"""

import xarray as xr



def main():
    ds_m=xr.open_dataset('../data/raw/reanalysis/omega_01_06_21.nc')
    ds_s=xr.open_dataset('../data/processed/netcdf/pressure_output.nc')
    print(ds_m)
    for time in ds_m['time'].values:
        ds_unit=ds_s.sel(time=time, method='nearest')
        print(ds_unit)
        
    


if __name__ == '__main__':
    main()
