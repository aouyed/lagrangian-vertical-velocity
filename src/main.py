
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
    frame0=np.squeeze(ds_unit[flow_var].values)
    frame0=np.nan_to_num(frame0)
    nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    files.pop(0)
    print(len(files))
 
    ds_total = xr.Dataset()
    for file in files:
        ds_unit=xr.open_dataset(file)
        date=ds_unit['time'].values
        print(date)
        #ds_unit, frame0=calc.calc(ds_unit,frame0,nframe0)
        print(ds_unit)
      
        if not ds_total:
            ds_total = ds_unit
        else:
            ds_total = xr.concat([ds_total, ds_unit], 'time')
    date= pd.to_datetime(str(date[0]))
    ds_total.to_netcdf(NC_PATH+'pressure_output.nc')
   
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
    animation.save(PLOT_PATH+ var+'_'+scatterv+'.gif')
    
def post_plots(ds):
    temp_var='cloud_top_temperature'
    ds['vel_error']=ds['height_vel']-ds['height_tendency']
    # calc.hist2d(ds, PLOT_PATH+ 'velheight', ['cloud_top_height','height_vel'], [0,10], [-0.25,0.25])
    # calc.hist2d(ds,PLOT_PATH+ 'veltemp', [temp_var,'height_vel'], [220,290], [-1,1])
    # calc.hist2d(ds, PLOT_PATH+ 'tendtemp', [temp_var,'height_tendency'], [220,290], [-1,1])
    # calc.hist2d(ds, PLOT_PATH+ 'speeddtemp', [temp_var,'speed'], [220,290], [-5,5])
    # calc.hist2d(ds, PLOT_PATH+ 'speedvel', ['height_vel','speed'], [-1,1], [0,5])
    # calc.hist2d(ds, PLOT_PATH+ 'speedtend', ['height_tendency','speed'], [-1,1], [0,5])
    # calc.hist2d(ds, PLOT_PATH+ 'veltend', ['height_tendency','height_vel'], [-0.5,0.5], [-0.25,0.25])
    # calc.hist2d(ds, PLOT_PATH+ 'accheight', ['cloud_top_height','height_acceleration'], [0,5], [-2e-2,2e-2])

    # calc.hist2d(ds, PLOT_PATH+ 'acctemp', [temp_var,'height_acceleration'], [220,290], [-1e-4,1e-4])
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()

    calc.scatter2d(ds, PLOT_PATH+ 'veltemp', [temp_var,'height_vel'], [200,240], [-1,1])
    calc.scatter2d(ds, PLOT_PATH+ 'velheight', ['cloud_top_height','height_vel'], [200,240], [-10,50])
    calc.scatter2d(ds,PLOT_PATH+  'tendtemp', [temp_var,'height_tendency'], [200,240], [-10,10])
    calc.scatter2d(ds,PLOT_PATH+  'tendheight', ['cloud_top_height','height_tendency'], [200,240], [-10,10])
    calc.scatter2d(ds, PLOT_PATH+ 'acctemp', [temp_var,'height_acceleration'], [200,240], [-1e-2,1e-2])
    calc.scatter2d(ds, PLOT_PATH+ 'accheight', ['cloud_top_height','height_acceleration'], [200,240], [-1e-2,1e-2])
    calc.scatter2d(ds, PLOT_PATH+ 'acctend', ['cloud_top_height','height_acceleration_e'], [200,240], [-1e-2,1e-2])

    calc.scatter2d(ds, PLOT_PATH+ 'speedheight', ['cloud_top_height','speed'], [200,240], [-1e-4,1e-4])
    calc.scatter2d(ds, PLOT_PATH+ 'speedvel', ['speed','height_vel'], [200,240], [-1e-4,1e-4])
    calc.scatter2d(ds, PLOT_PATH+ 'speedtend', ['speed','height_tendency'], [200,240], [-1e-4,1e-4])
    calc.scatter2d(ds, PLOT_PATH+ 'moistvel', ['belwp','height_vel'], [200,240], [-10,10])
    calc.scatter2d(ds, PLOT_PATH+ 'moisl', ['belwp','height_vel'], [200,240], [-10,10])

    calc.scatter2d(ds, PLOT_PATH+ 'moistheight', ['belwp','cloud_top_height'], [200,240], [-10,10])

    


def analysis(ds):
    ds=ds.astype(np.float32)
    print(ds['time'].values)

    #ds=ds.coarsen(lat=100, boundary='trim').mean().coarsen(lon=100, boundary='trim').mean()
    ds=ds.coarsen(time=3,boundary='trim').mean()
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    print(ds['speed'].mean())
    ds['height_acceleration']=ds['height_vel'].diff('time')/1800
    ds['height_acceleration_e']=ds['height_tendency'].diff('time')/1800    

    post_plots(ds)

    
    #plot_loop(ds, 'height_acceleration', calc.quiver_hybrid, -0.0001, 0.0001,'RdBu')
    #plot_loop(ds, 'height_acceleration_e', calc.quiver_hybrid, -0.0001, 0.0001,'RdBu')
    plot_loop(ds, 'temperature_ir', calc.quiver_hybrid, 220, 290,'viridis','_an')
    plot_loop(ds, 'height_vel', calc.quiver_hybrid, -1, 1,'RdBu','height_vel_an')
    plot_loop(ds, 'height_tendency', calc.quiver_hybrid, -1,1,'RdBu', 'height_tendency_an')
    
    

def main():
    ds= preprocessing()
    #ds=xr.open_dataset(NC_PATH+'01-06-2021-23:38:20_output.nc')
    #analysis(ds)

if __name__ == '__main__':
    main()
