#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:34:55 2024

@author: amirouyed
"""

from parameters import parameters
import matplotlib.pyplot as plt
import numpy as np

param=parameters()


param.var_label='flagged_diff_d'+ param.var
diff_df=np.load(param.var_pathname(param.start_date))
param.var_label='flagged_diff_dthresh'
diff_df_thresh=np.load(param.var_pathname(param.start_date))
plt.scatter(diff_df_thresh.ravel(), diff_df.ravel())
