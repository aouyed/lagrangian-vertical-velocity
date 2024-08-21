#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:33:47 2024

@author: amirouyed
"""

from datetime import datetime 
from datetime import timedelta
import pandas as pd 
import numpy as np
import glob



class parameters:
    START_Y=1700
    END_Y=1900
    START_X=1500
    END_X=2500
    
    def __init__(self):
      
      
        self.prefix='OR_ABI-L2-ACHTF-M6_G18'
        self.Lambda=0.07
        self.starty=1700
        self.frame_slice=np.index_exp[1700:1900, 1500:2500]
        self.start_date=datetime(2023,7,1,18,0)
        self.end_date=datetime(2023,7,1,23,40)
        self.var_label='flagged_overlap_dthresh'
        self.processed_path='../data/processed/'
        self.raw_path='../data/raw/'
        self.temp_thresh=250
        self.var='TEMP'
        self.amv_var_label='flagged_amv'
        self.dt=timedelta(minutes=10)

        
    
    
    def var_pathname(self, date):
        string=self.processed_path+'l'+str(self.Lambda)+'_'+date.strftime(self.var_label+'_%Y%m%d%H%M.npy')
        return string
    
    def overlap_gif_pathname(self):
        string=self.processed_path+'plots/l'+str(self.Lambda)+self.var_label+'_overlap.gif'
        return string
    
    def var_gif_pathname(self):
        string=self.processed_path+'plots/l'+str(self.Lambda)+'_'+self.var+'.gif'
        return string
    
    def amv_gif_pathname(self):
        string=self.processed_path+'plots/l'+str(self.Lambda)+'_quiver.gif'
        return string 
    
    def amv_pathname(self, date):
        
        amv_file=date.strftime(self.amv_var_label+'_%Y%m%d%H%M.npy')
        string= self.processed_path+'l'+str(self.Lambda)+'_' +amv_file
        return string

 
    def date_string(self, date):
        d0=datetime(date.year, 1,1)
        delta= date-d0
        day_string=str(round(delta.days)+1)
        date_string=self.prefix+'_'+date.strftime('s%Y')+day_string+date.strftime('%H%M')
        filename=glob.glob('../data/raw/'+date_string+'*')
        assert len(filename)==1, 'ill defined filename'
        return filename[0]

    def calc_datelist(self):
        datelist=pd.date_range(self.start_date, self.end_date, freq='10min')
        return datelist