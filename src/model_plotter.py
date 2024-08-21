#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 19:26:46 2021

@author: aouyed
"""

import xarray as xr
import calculators as calc
import main


ds=xr.open_dataset('../data/processed/model.nc')

#ds=ds.coarsen(time=3,boundary='trim').mean()
ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
ds['z_acceleration']=ds['w'].diff('time')/3600

calc.scatter2d(ds, main.PLOT_PATH+ 'model', ['cloud_top_height','w'], [0,10], [-0.25,0.25])
calc.scatter2d(ds, main.PLOT_PATH+ 'modela', ['cloud_top_height','z_acceleration'], [0,10], [-1e-4,1e-4])
    