
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
from pylab import MaxNLocator


PLOT_PATH='../data/processed/plots/'
NC_PATH='../data/processed/netcdf/'
flow_var='cloud_top_pressure'
DATE_FORMAT="%m-%d-%Y-%H:%M:%S"
FOLDER='01_28_21'
#FOLDER='05_30_21'
#FOLDER='may'


def preprocessing():
    files=natsorted(glob.glob('../data/interim/'+FOLDER+'/*'))
    print(len(files))
    ds_unit=xr.open_dataset(files[0]) 
    frame0=np.squeeze(ds_unit[flow_var].values)
    frame0=np.nan_to_num(frame0)
    height0=np.squeeze(ds_unit['cloud_top_height'].values)
    pressure0=np.squeeze(ds_unit['cloud_top_pressure'].values)
    files.pop(0)
    print(len(files))
 
    ds_total = xr.Dataset()
    for file in files:
        ds_unit=xr.open_dataset(file)
        date=ds_unit['time'].values
        print(date)
        ds_unit, frame0, height0, pressure0 =calc.calc(ds_unit,frame0,pressure0,height0)
        print(ds_unit)
      
        if not ds_total:
            ds_total = ds_unit
        else:
            ds_total = xr.concat([ds_total, ds_unit], 'time')
    date= pd.to_datetime(str(date[0]))
    ds_total.to_netcdf(NC_PATH+FOLDER+'_output.nc')
   
    return ds_total

def plot_loop(ds, var, func, vmin, vmax, cmap,tag):
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
    if func != calc.marginal_an:    
        #cbar=plt.colorbar(im)
        fig.colorbar(im, orientation="horizontal", pad=0.2)

    animation = camera.animate()
    animation.save(PLOT_PATH+ var+'_'+tag+'.gif')
    
def post_plots(ds):
    temp_var='cloud_top_temperature'
    print(abs(ds['vel_error']).mean())
    print(abs(ds['height_vel']).mean())
    print(abs(ds['height_tendency']).mean())
 
   
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()

    calc.scatter2d(ds, PLOT_PATH+ 'veltemp', [temp_var,'height_vel'], [200,240], [-1,1])
    calc.scatter2d(ds, PLOT_PATH+ 'velheight', ['cloud_top_height','height_vel'], [200,240], [-2,2])
    calc.scatter2d(ds, PLOT_PATH+ 'vel_error', ['cloud_top_height','vel_error'], [200,240], [-2,2])
    calc.scatter2d(ds,PLOT_PATH+  'tendtemp', [temp_var,'height_tendency'], [200,240], [-10,10])
    calc.scatter2d(ds,PLOT_PATH+  'tendheight', ['cloud_top_height','height_tendency'], [200,240], [-2,2])
    calc.scatter2d(ds, PLOT_PATH+ 'acctemp', [temp_var,'height_acceleration'], [200,240], [-1e-2,1e-2])
    calc.scatter2d(ds, PLOT_PATH+ 'accheight', ['cloud_top_height','height_acceleration'], [200,240], [-1e-2,1e-2])
    calc.scatter2d(ds, PLOT_PATH+ 'acctend', ['cloud_top_height','height_acceleration_e'], [200,240], [-1e-2,1e-2])
    calc.scatter2d(ds, PLOT_PATH+ 'moistvel', ['belwp','height_vel'], [200,240], [-10,10])
    calc.scatter2d(ds, PLOT_PATH+ 'moisl', ['belwp','height_vel'], [200,240], [-10,10])

    calc.scatter2d(ds, PLOT_PATH+ 'moistheight', ['belwp','cloud_top_height'], [200,240], [-10,10])
    calc.scatter2d(ds, PLOT_PATH+ 'pvelheight', ['cloud_top_height','pressure_vel'], [200,240], [-2,2])
    calc.scatter2d(ds, PLOT_PATH+ 'ptendency', ['cloud_top_height','pressure_tendency'], [200,240], [-2,2])


def analysis(ds):
    print(ds['time'].values)
    #ds=ds.coarsen(time=3,boundary='trim').mean()
    ds=ds.sel(time=ds['time'].values[18])
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
    ds=ds.where(ds['cloud_top_pressure']<700)
    ds['pressure_vel']=100*ds['pressure_vel']
    ds['pressure_tendency']=100*ds['pressure_tendency']
    ds['pressure_acceleration']=ds['pressure_vel'].diff('time')/1800
    #ds['height_acceleration_e']=ds['height_tendency'].diff('time')/1800    
    ds['vel_error']=ds['height_vel']-ds['height_tendency']
       
    calc.map_plotter(ds, 'pressure_vel','pressure_vel', units_label='cm/s', vmin=-0.2, vmax=0.2)
    calc.map_plotter(ds, 'pressure_acceleration','pressure_acceleration', units_label='cm/s', vmin=-0.0002, vmax=0.0002)

    calc.map_plotter(ds, 'cloud_top_pressure','cloud_top_pressure', units_label='hpa')

   
    
    
    ds=ds.where(ds['cloud_top_pressure']>850)
   
    
    

def main():
    ds= preprocessing()
    ds=xr.open_dataset(NC_PATH+FOLDER+'_output.nc')
    m.plot_loop(ds, 'cloud_top_pressure',calc.implot_quiver, 200, 1000,'winter',m.FOLDER)
    m.plot_loop(ds, 'pressure_vel',calc.implot_quiver, -10, 10,'RdBu',m.FOLDER)
    m.plot_loop(ds, 'pressure_tendency',calc.implot_quiver,-10, 10,'RdBu',m.FOLDER)

if __name__ == '__main__':
    main()
