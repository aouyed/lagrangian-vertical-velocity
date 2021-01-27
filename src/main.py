
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

PLOT_PATH='../data/processed/plots/'
NC_PATH='../data/processed/netcdf/'
flow_var='temperature_ir'
DATE_FORMAT="%m-%d-%Y-%H:%M:%S"



def preprocessing():
    files=natsorted(glob.glob('../data/interim/01_06/*'))
    print(len(files))
    ds_unit=xr.open_dataset(files[0])
    #ds_unit[flow_var].plot.imshow(vmin=0, vmax=10)
    #plt.savefig(flow_var+'0.png')
  
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
        
        ds_unit, frame0=calc.calc(ds_unit,frame0)
      
        if not ds_total:
            ds_total = ds_unit
        else:
            ds_total = xr.concat([ds_total, ds_unit], 'time')
    #ds_total=ds_total.reindex(image_y=list(reversed(ds_total['image_y'])))
    print(ds_total)
    ds_total.to_netcdf(NC_PATH+date[0].strftime(DATE_FORMAT)+'_output.nc')
   
    return ds_total

def plot_loop(ds, var, func, vmin, vmax):
    fig, ax = plt.subplots(dpi=300)
    camera = Camera(fig)
    dates=ds['time'].values
    switch=True
    for date in dates:
        print(date)
        ds_unit=ds.sel(time=date)
        #calc.quick_plotter(ds_unit, date)
        ax, fig, im =func(ds_unit, ds_unit[var].values, vmin,vmax,date, ax, fig)
        camera.snap()
    cbar=plt.colorbar(im)
    animation = camera.animate()
    animation.save(var+'.gif')
def main():
    ds= preprocessing()
    ds=xr.open_dataset(NC_PATH+'01-06-2021-23:38:20_output.nc')
    plot_loop(ds, flow_var, calc.quiver_hybrid, 230, 290)
    #plot_loop(ds, 'height_tendency')

if __name__ == '__main__':
    main()
