
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as rgi
import numpy as np
#import metpy.calc
from metpy.units import units
import cv2
from dateutil import parser
import glob
from natsort import natsorted
from matplotlib import animation
import calculators as calc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from celluloid import Camera
import pandas as pd

PLOT_PATH='../data/processed/plots/'
NC_PATH='../data/processed/netcdf/'
flow_var='temperature_ir'
DATE_FORMAT="%m-%d-%Y-%H:%M:%S"



def preprocessing():
    files=natsorted(glob.glob('../data/interim/01_06/*'))
    print(len(files))
    ds_unit=xr.open_dataset(files[0]) 
    ds_unit=ds_unit.coarsen(lat=100, boundary='trim').mean().coarsen(lon=100, boundary='trim').mean()
    frame0=np.squeeze(ds_unit[flow_var].values)
    frame0=np.nan_to_num(frame0)
    nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    files.pop(0)
    print(len(files))
 
    ds_total = xr.Dataset()
    for file in files:
        ds_unit=xr.open_dataset(file)
        ds_unit=ds_unit.coarsen(lat=100, boundary='trim').mean().coarsen(lon=100, boundary='trim').mean()
        date=ds_unit['time'].values
        print(date)
        ds_unit, frame0=calc.calc(ds_unit,frame0)
        print(ds_unit)
      
        if not ds_total:
            ds_total = ds_unit
        else:
            ds_total = xr.concat([ds_total, ds_unit], 'time')
    date= pd.to_datetime(str(date[0]))
    ds_total.to_netcdf(NC_PATH+date.strftime(DATE_FORMAT)+'_output.nc')
   
    return ds_total

def plot_loop(ds, var, func, vmin, vmax, cmap,scatterv):
    fig, ax = plt.subplots(dpi=300)
    camera = Camera(fig)
    dates=ds['time'].values
    for date in dates:
        print(date)
        ds_unit=ds.sel(time=date)
        ax, fig, im =func(ds_unit, ds_unit[var].values, vmin,vmax,date, ax, fig, cmap,scatterv)
        camera.snap()
    cbar=plt.colorbar(im)
    animation = camera.animate()
    animation.save(var+'_'+scatterv+'.gif')
def main():
    #ds= preprocessing()
    ds=xr.open_dataset(NC_PATH+'01-06-2021-23:38:20_output.nc')
    ds=ds.astype(np.float32)
    #ds=ds.coarsen(lat=100, boundary='trim').mean().coarsen(lon=100, boundary='trim').mean()
    ds=ds.coarsen(time=3,boundary='trim').mean()
    ds['height_acceleration']=ds['height_vel'].diff('time')/1800
    ds['height_acceleration_e']=ds['height_tendency'].diff('time')/1800
    ds['vel_error']=ds['height_vel']-ds['height_tendency']
    plot_loop(ds, 'temperature_ir', calc.scatter_hybrid, 230, 290,'viridis','height_acceleration')
    plot_loop(ds, 'temperature_ir', calc.scatter_hybrid, 230, 290,'viridis','vel_error')
    plot_loop(ds, 'temperature_ir', calc.scatter_hybrid, 230, 290,'viridis','height_vel')
    plot_loop(ds, 'temperature_ir', calc.scatter_hybrid, 230, 290,'viridis','height_tendency')
    
    #plot_loop(ds, 'height_acceleration', calc.quiver_hybrid, -0.0001, 0.0001,'RdBu')
    #plot_loop(ds, 'height_acceleration_e', calc.quiver_hybrid, -0.0001, 0.0001,'RdBu')
    #plot_loop(ds, 'height_vel', calc.quiver_hybrid, -1, 1,'RdBu')
    #plot_loop(ds, 'height_tendency', calc.quiver_hybrid, -1,1,'RdBu')
    

if __name__ == '__main__':
    main()
