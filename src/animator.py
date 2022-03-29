#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:19:38 2022

@author: aouyed
"""

def animation(ds_total, tag):
    m.plot_loop(ds_total, 'divergence_mean', c.quiver_hybrid, -10, 10,'RdBu',m.FOLDER+tag)    

    # cmap = c.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)
    # m.plot_loop(ds_total, 'cloud_top_pressure_mean', c.implot, 0, 1000,'viridis',m.FOLDER+tag)
    # m.plot_loop(ds_total, 'id_map', c.implot, 0, 1000,cmap,m.FOLDER + tag)
    # m.plot_loop(ds_total, 'divergence_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
    # m.plot_loop(ds_total, 'vorticity_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
    # # m.plot_loop(ds_total, 'size_rate_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
    # # m.plot_loop(ds_total, 'pressure_adv_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
    # m.plot_loop(ds_total, 'pressure_vel_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
    # m.plot_loop(ds_total, 'pressure_tendency_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
    # m.plot_loop(ds_total, 'pressure_rate_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+ tag)
    # # m.plot_loop(ds_total, 'thresh_map', c.implot, 0, 255,'viridis',m.FOLDER + tag)
    # # m.plot_loop(ds_total, 'size_map', c.implot, 0, 100,'viridis',m.FOLDER + tag)
    # m.plot_loop(ds_total, 'pressure_vel', c.implot, -2, 2,'RdBu',m.FOLDER + tag)
    # m.plot_loop(ds_total, 'pressure_tendency', c.implot, -2, 2,'RdBu',m.FOLDER + tag)
    # m.plot_loop(ds_total, 'cloud_top_pressure', c.implot, 0, 1000,'viridis',m.FOLDER + tag)

