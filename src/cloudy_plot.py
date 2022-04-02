
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
from celluloid import Camera


class cloudy_plot:
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
    def __init__(self,clouds):
        self.clouds=clouds
        
  
    def quiver_hybrid(self, ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
        ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
        ax, fig, im=self.implot(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv)
        #ds=ds.coarsen(lat=3, boundary='trim').mean().coarsen(lon=3, boundary='trim').mean()
        X,Y=np.meshgrid(ds['lon'].values,ds['lat'].values)
        Q = ax.quiver(X,Y, np.squeeze(ds['u'].values), np.squeeze(ds['v'].values))
        return   ax, fig, im
        
    def implot(self, ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color='grey')
        im = ax.imshow(values, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, 
                       extent=[ds['lon'].min().item(),
                               ds['lon'].max().item(),ds['lat'].min().item(),ds['lat'].max().item()])
        return ax, fig, im
        
    def time_series_plotter(ds, label, tag):
        fig, ax= plt.subplots()
        #for idno in ds['id'].values:
    
        for idno in ds['id'].values:
            ds_unit=ds.sel(id=idno)
            dates=ds_unit['time'].values
            dates=pd.to_datetime(dates).hour
            ax.plot(dates,ds_unit[label].values, label=str(idno))
        #ax.legend()
        #ax.set_ylim(0,150)
        ax.set_xlabel('hour')
        ax.set_ylabel('cloud top pressure')
    
        plt.savefig('../data/processed/plots/ts_'+label+tag+'.png', dpi=300)
        plt.show()
        plt.close()

    
    def plot_loop(self, ds, var, func, vmin, vmax, cmap,tag):
        fig, ax = plt.subplots(dpi=300)
        camera = Camera(fig)
        dates=ds['time'].values
        for date in dates:
            print(date)
            ds_unit=ds.sel(time=date)
            values=ds_unit[var].values
            values[values==0]=np.nan
            ax, fig, im =func(ds_unit, values, vmin,vmax,date, ax, fig, cmap, tag)
            ax.text(0.5, 1.01, np.datetime_as_string(date, timezone='UTC'),
                    transform=ax.transAxes)
    
            camera.snap()
    
        animation = camera.animate()
        animation.save(config.PLOT_PATH+ var+'_'+tag+'.gif')
    
    
    def animate(self, tag):
        ds_total=self.clouds.ds_clouds_mean
        ds_total=ds_total.sel(lat=slice(0,25), lon=slice(-100, -75))
        self.plot_loop(ds_total, 'divergence_mean',self.quiver_hybrid, -10, 10,'RdBu',config.FOLDER+tag)    
    
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
            
            
    
   