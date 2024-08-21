#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:45:20 2024

@author: amirouyed
"""

import xarray as xr
from parameters import parameters
param=parameters()
ds=xr.open_dataset('../data/raw/OR_ABI-L2-DMWF-M6C14_G18_s20231822100222_e20231822109530_c20231822123540.nc')
breakpoint()