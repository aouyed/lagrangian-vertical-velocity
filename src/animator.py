from datetime import datetime
from datetime import timedelta
import pandas as pd 
import glob
import matplotlib.pyplot as plt
from celluloid import Camera
import xarray as xr
import numpy as np
from tqdm import tqdm
import amv_calculators as ac
from netCDF4 import Dataset
import cartopy.crs as ccrs
from parameters import parameters

def implot(values, lat_grid, lon_grid, date,ax, fig, cmap, climits, vmin, vmax):
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    im = plt.imshow(values, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)
    return ax, fig, im

def cartopy_pmesh(values, lat_grid, lon_grid, date, ax , fig, cmap, climits, vmin, vmax):
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    if climits==False:
        im=plt.pcolormesh(lon_grid, lat_grid, values, cmap=cmap)
    else:
        im=plt.pcolormesh(lon_grid, lat_grid, values, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.coastlines()
    gls=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gls.top_labels=False 

    return  ax, fig, im

def quiver_plot(values, lat_grid, lon_grid, date, ax , fig, cmap, climits, vmin, vmax):
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    if climits==False:
        im=plt.pcolormesh(lon_grid, lat_grid, values, cmap=cmap)
    else:
        im=plt.pcolormesh(lon_grid, lat_grid, values, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.coastlines()
    gls=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gls.top_labels=False 

    return  ax, fig, im

    

def var_plot_loop(param, func, cmap,  units='K', climits=False, vmin=200,vmax=300):
    fig=plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())

    

    camera = Camera(fig)
    for date in tqdm(param.calc_datelist()):

        filename=param.date_string(date)
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        
        ds=xr.open_dataset(filename)
        values=ds[param.var].values
        values=values[param.frame_slice]
        values[values>param.temp_thresh]=np.nan

        abi_lat=abi_lat[param.frame_slice]
        abi_lon=abi_lon[param.frame_slice]
        
        ax, fig, im =func(values, abi_lat, abi_lon, date,ax, fig, cmap, climits, vmin, vmax)
        ax.coastlines()

        ax.text(0.5, 1.01, str(date),transform=ax.transAxes)
        #cb=plt.colorbar(im,ax=ax)
        camera.snap()
    cbar=plt.colorbar(location='bottom', label=units)
    animation = camera.animate()
    animation.save(param.var_gif_pathname())
    
    

def overlap_plot_loop(param, func, cmap,  units='K', climits=False, vmin=200,vmax=300):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)

    ax = plt.axes(projection=ccrs.PlateCarree())
    

    camera = Camera(fig)
    for date in tqdm(param.calc_datelist()[:2]):
        filename=param.date_string(date)
        
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        
        print(param.var_pathname(date))
        values=np.load(param.var_pathname(date))

        abi_lat=abi_lat[param.frame_slice]
        abi_lon=abi_lon[param.frame_slice]

        
        ax, fig, im =func(values, abi_lat, abi_lon, date,ax, fig, cmap, climits, vmin, vmax)
        #ax.coastlines()

        ax.text(0.5, 1.01, str(date),transform=ax.transAxes)
        #cb=plt.colorbar(im,ax=ax)
        camera.snap()
    cbar=plt.colorbar(location='bottom', label=units)
    animation = camera.animate()
    animation.save(param.overlap_gif_pathname())
    
        
        
def flow_ds(x,y, flowx, flowy):
    dims = ('x', 'y')

    coords = {
          'x': x,
          'y': y, 
      }
    
    amv_ds = xr.Dataset(
          {'flowx': (dims, flowx)
           ,'flowy': (dims, flowy)},
          coords=coords)
    return amv_ds

def quiver_loop(param, func, cmap,  units='K', climits=False, vmin=200,vmax=300):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)

    ax = plt.axes(projection=ccrs.PlateCarree())
    

    camera = Camera(fig)
    for date in tqdm(param.calc_datelist()[:2]):

        filename=param.date_string(date)       
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        flowd=np.load(param.amv_path_name(date))
        
        flowx=flowd[:,:,0]
        flowy=flowd[:,:,1]
    
        ds=xr.open_dataset(filename)
        values=ds[param.var].values
        #values=abi_lat
        values[values==0]=np.nan
        values=values[param.frame_slice]
        values[values>param.temp_thresh]=np.nan

        abi_lat=abi_lat[param.frame_slice]
        abi_lon=abi_lon[param.frame_slice]
        ds_amv=xr.Dataset({'flowx':(['y','x'],flowx),'flowy':(['y','x'],flowy), 'lat':(['y','x'],abi_lat),'lon':(['y','x'],abi_lon)})
        ds_amv=ds_amv.coarsen(x=20, y=20, boundary='trim').mean()
        
        ax, fig, im =func(values, abi_lat, abi_lon, date,ax, fig, cmap, climits, vmin, vmax)
        plt.quiver(ds_amv['lon'].values,ds_amv['lat'].values, ds_amv['flowx'].values, ds_amv['flowy'].values, scale=100, color='red')        
        #ax.coastlines()

        ax.text(0.5, 1.01, str(date),transform=ax.transAxes)
        #cb=plt.colorbar(im,ax=ax)
        camera.snap()

    #camera.snap()
    cbar=plt.colorbar(location='bottom', label=units)
    animation = camera.animate()
    animation.save(param.amv_gif_pathname())
    

def main(param):
 
    #var_plot_loop(param, cartopy_pmesh, 'viridis', units='', climits=True)
    #quiver_loop(param, cartopy_pmesh, 'viridis', units='K', climits=True)
    overlap_plot_loop( param, cartopy_pmesh, 'viridis', units='K',  climits=True, vmin=-1, vmax=2)

    # var_label='warped_dthresh'

   # quiver_loop(param, cartopy_pmesh, 'viridis', units='K', climits=True)

    #quiver_loop(param)

    #prefix='OR_ABI-L1b-RadF-M6C14_G18'
    #plot_loop(prefix, datelist, 'Rad', cartopy_pmesh, 'viridis', 'test_rad.png', units='', climits=True)

    
if __name__=='__main__':
    param=parameters()
    main(param)
