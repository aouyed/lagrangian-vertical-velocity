#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:34:55 2024

@author: amirouyed
"""

from parameters import parameters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
param=parameters()

param=parameters()
param.prefix='OR_ABI-L2-ACHA2KMF-M6_G18'
param.var='HT'

param.Lambda=0.07
param.var_label='flagged_diff_d'+ param.var
diff_df=np.load(param.var_pathname(param.start_date))
param.var_label='flagged_diff_dthresh'
diff_df_thresh=np.load(param.var_pathname(param.start_date))
param.var_label='flagged_warped_dthresh'
warped_dthresh=np.load(param.var_pathname(param.start_date))
param.var_label='flagged_dthresh'
dthresh=np.load(param.var_pathname(param.start_date))

param.var_label='flagged_d'+ param.var
dHT=np.load(param.var_pathname(param.start_date))
param.var_label='flagged_warped_d'+ param.var

dHT_warped=np.load(param.var_pathname(param.start_date))


data={'diff_thresh': diff_df_thresh.ravel(), 'diff_df': diff_df.ravel()}

df=pd.DataFrame(data)
df=df.dropna()

#plt.scatter(df['diff_thresh'].values, df['diff_df'].values )
plt.imshow(diff_df, cmap='viridis')
plt.colorbar()
plt.show()
plt.close()
plt.hist(diff_df.ravel())
plt.show()
plt.close()
plt.hist(dHT.ravel())
plt.show()
plt.close()
plt.hist(dHT_warped.ravel())
plt.show()
plt.close()
