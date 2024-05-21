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


def implot(values, lat_grid, lon_grid, date,ax, fig, cmap):
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    im = ax.imshow(values, cmap=cmap, origin='upper')
    return ax, fig, im

def cartopy_pmesh(values, lat_grid, lon_grid, date, ax , fig, cmap):
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    im=ax.pcolormesh(lon_grid, lat_grid, values, cmap=cmap)
    ax.coastlines()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    return  ax, fig, im

    

def plot_loop(prefix, datelist, var, func, cmap):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())

    camera = Camera(fig)
    for date in tqdm(datelist[:2]):
        filename=date_string(prefix, date)
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        
        ds=xr.open_dataset(filename)
        values=ds[var].values
        #values=abi_lat
        values[values==0]=np.nan
        values=values[1700:1900, 1500:2500]
        abi_lat=abi_lat[1700:1900, 1500:2500]
        abi_lon=abi_lon[1700:1900, 1500:2500]

        
        ax, fig, im =func(values, abi_lat, abi_lon, date,ax, fig, cmap)
        #ax.coastlines()

        ax.text(0.5, 1.01, str(date),transform=ax.transAxes)

        camera.snap()

    animation = camera.animate()
    animation.save('../data/processed/test.gif')
    
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
    end_date=datetime(2023,7,1,23,50)
    datelist=pd.date_range(start_date, end_date, freq='10min')
    
    
    #datelist = pd.date_range(start=start_date,end=end_date,freq='10m')
    prefix='OR_ABI-L2-ACHTF-M6_G18'
    plot_loop(prefix, datelist, 'TEMP', cartopy_pmesh, 'viridis')
    
if __name__=='__main__':
    main()
