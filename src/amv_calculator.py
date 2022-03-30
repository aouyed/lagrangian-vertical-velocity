
import cloud_calculators as cc
import main as m 
import cv2
import numpy as np
import xarray as xr
import pandas as pd
import main as m 
import calculators as c
import kinematics_tracker as kt
import config as config 
from centroidtracker import CentroidTracker
import main as m 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 
from scipy import stats
import animator


class amv_calculator:
    """An object that tracks and tags clouds, and calculates kinematic 
    and thermodynamic properties `.
    Parameters
    ----------
    file : string, 
        Input file with calculated AMVs.
    
    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated robust location.
    
    """
    def __init__(self,file):
        self.file=file
        self.object_tracker()
        
    def preprocessing():
        files=natsorted(glob.glob('../data/interim/'+FOLDER+'/*'))
        ds_total = xr.Dataset()
        for file in files:
            ds_unit=xr.open_dataset(file)
            if not ds_total:
                ds_total = ds_unit
            else:
                ds_total = xr.concat([ds_total, ds_unit], 'time')

        ds_total.to_netcdf(NC_PATH+FOLDER+'_output.nc')
   
    return ds_total