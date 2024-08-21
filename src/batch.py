#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:25:49 2024

@author: amirouyed
"""


from parameters import parameters
import animator
import calculators
import amv_calculators
from datetime import datetime

param=parameters()
#param.end_date=datetime(2023,7,1,18,10)

#param.Lambda=0.05
#amv_calculators.main(param)
param.prefix='OR_ABI-L2-ACHA2KMF-M6_G18'
param.var='HT'
calculators.main(param)
animator.main(param)
