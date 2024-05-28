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

    

def plot_loop(prefix, datelist, var, func, cmap, filename, units='K', climits=False, vmin=200,vmax=300):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)

    #ax = plt.axes(projection=ccrs.PlateCarree())
    

    camera = Camera(fig)
    for date in tqdm(datelist):

        filename=date_string(prefix, date)
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        
        ds=xr.open_dataset(filename)
        values=ds[var].values
        plt.imshow(values, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)

        #values=abi_lat
        #values[values==0]=np.nan
        #values=values[1700:1900, 1500:2500]
        #abi_lat=abi_lat[1700:1900, 1500:2500]
        #abi_lon=abi_lon[1700:1900, 1500:2500]

        
        #ax, fig, im =func(values, abi_lat, abi_lon, date,ax, fig, cmap, climits, vmin, vmax)
        #ax.coastlines()

        #ax.text(0.5, 1.01, str(date),transform=ax.transAxes)
        #cb=plt.colorbar(im,ax=ax)
        camera.snap()
    #cbar=plt.colorbar(location='bottom', label=units)
    animation = camera.animate()
    animation.save('../data/processed/test_complete.gif')
    
        
        
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

def quiver_loop(prefix, datelist, var, func, cmap, filename, units='K', climits=False, vmin=200,vmax=300):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)

    ax = plt.axes(projection=ccrs.PlateCarree())
    

    camera = Camera(fig)
    print(datelist)
    for date in tqdm(datelist):

        filename=date_string(prefix, date)
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        amv_file=date.strftime('vis_amv_%Y%m%d%H%M.npy')
        flowd=np.load('../data/processed/'+amv_file)
        
        flowx=flowd[:,:,0]
        flowy=flowd[:,:,1]
    
        ds=xr.open_dataset(filename)
        values=ds[var].values
        #values=abi_lat
        values[values==0]=np.nan
        values=values[1700:1900, 1500:2500]
        abi_lat=abi_lat[1700:1900, 1500:2500]
        abi_lon=abi_lon[1700:1900, 1500:2500]
        ds_amv=xr.Dataset({'flowx':(['y','x'],flowx),'flowy':(['y','x'],flowy), 'lat':(['y','x'],abi_lat),'lon':(['y','x'],abi_lon)})
        ds_amv=ds_amv.coarsen(x=20, y=20, boundary='trim').mean(_)
        
        ax, fig, im =func(values, abi_lat, abi_lon, date,ax, fig, cmap, climits, vmin, vmax)
        plt.quiver(ds_amv['lon'].values,ds_amv['lat'].values, ds_amv['flowx'].values, ds_amv['flowy'].values, scale=100, color='red')        
        #ax.coastlines()

        ax.text(0.5, 1.01, str(date),transform=ax.transAxes)
        #cb=plt.colorbar(im,ax=ax)
        camera.snap()

    #camera.snap()
    cbar=plt.colorbar(location='bottom', label=units)
    animation = camera.animate()
    animation.save('../data/processed/test_cbar.gif')
    
def date_string(prefix, date):
    d0=datetime(date.year, 1,1)
    delta= date-d0
    day_string=str(round(delta.days)+1)
    date_string=prefix+'_'+date.strftime('s%Y')+day_string+date.strftime('%H%M')
    filename=glob.glob('../data/raw/'+date_string+'*')
    assert len(filename)==1, 'ill defined filename'
    return filename[0]
    
def main():
    start_date=datetime(2023,7,1,18,0)
    end_date=datetime(2023,7,1,23,40)
    datelist=pd.date_range(start_date, end_date, freq='10min')
    #datelist = pd.date_range(start=start_date,end=end_date,freq='10m')
    prefix='OR_ABI-L2-ACHTF-M6_G18'
    quiver_loop(prefix, datelist, 'TEMP', cartopy_pmesh, 'viridis', 'cbar.png', units='K', climits=True)

    #plot_loop(prefix, datelist, 'TEMP', implot, 'viridis', 'cbar.png', units='K', climits=True)
    #quiver_loop(pr)

    #prefix='OR_ABI-L1b-RadF-M6C14_G18'
    #plot_loop(prefix, datelist, 'Rad', cartopy_pmesh, 'viridis', 'test_rad.png', units='', climits=True)

    
if __name__=='__main__':
    print('hello')
    main()
