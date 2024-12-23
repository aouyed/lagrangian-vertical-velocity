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


def temp_filter(values,date):
    param_temp=parameters()
    param_temp.prefix='OR_ABI-L2-ACHTF-M6_G18'
    param_temp.var='TEMP'
    filename=param_temp.date_string(date)
    ds=xr.open_dataset(filename)
    values_temp=ds[param_temp.var].values
    values_temp=values_temp[param_temp.frame_slice]
    values[values_temp>param_temp.temp_thresh]=np.nan
    return values 
    

    

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
        values=temp_filter(values,date)
        
       # values[values>param.temp_thresh]=np.nan

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
    
    

def hist_plot_loop(param, func, cmap, title, units='K', climits=False, vmin=200,vmax=300):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)

    ax = plt.axes(projection=ccrs.PlateCarree())
    

    camera = Camera(fig)
    for date in tqdm(param.calc_datelist()):
        param.var_label='flagged_diff_d'+ param.var
        diff_df=np.load(param.var_pathname(date))
        param.var_label='flagged_diff_dthresh'
        diff_df_thresh=np.load(param.var_pathname(date))
        param.var_label='flagged_warped_dthresh'
        warped_dthresh=np.load(param.var_pathname(date))
        param.var_label='flagged_dthresh'
        dthresh=np.load(param.var_pathname(date))
        
        param.var_label='flagged_d'+ param.var
        dHT=np.load(param.var_pathname(date))
        param.var_label='flagged_warped_d'+ param.var
        
        dHT_warped=np.load(param.var_pathname(date))


        data={'diff_thresh': diff_df_thresh.ravel(), 'diff_df': diff_df.ravel()}
       
        df=pd.DataFrame(data)
        df=df.dropna()

       #plt.scatter(df['diff_thresh'].values, df['diff_df'].values )
        plt.imshow(diff_df, cmap='viridis')
        plt.colorbar()
        plt.show()
        plt.close()
        plt.hist(diff_df.ravel())
        plt.show()
        plt.close()
        plt.hist(dHT.ravel())
        plt.show()
        plt.close()
        plt.hist(dHT_warped.ravel())
        plt.show()
        plt.close()

        #cb=plt.colorbar(im,ax=ax)
        camera.snap()
    cbar=plt.colorbar(location='bottom', label=units)
    title=title+'_l'+str(param.Lambda)
    plt.title(title)
    animation = camera.animate()
    animation.save(param.overlap_gif_pathname())
    

def overlap_plot_loop(param, func, cmap, title, units='K', climits=False, vmin=200,vmax=300):
    #fig, ax = plt.subplots(dpi=300)
    fig=plt.figure(dpi=300)

    ax = plt.axes(projection=ccrs.PlateCarree())
    

    camera = Camera(fig)
    for date in tqdm(param.calc_datelist()):
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
    title=title+'_l'+str(param.Lambda)
    plt.title(title)
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
    for date in tqdm(param.calc_datelist()):

        filename=param.date_string(date)       
        file_id = Dataset(filename)
        abi_lat, abi_lon = ac.calculate_degrees(file_id)
        flowd=np.load(param.amv_pathname(date))
        
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
    #cbar=plt.colorbar(location='bottom', label=units)
    animation = camera.animate()
    animation.save(param.amv_gif_pathname())
    

def main(param):
 
    print('var_plot')
    
    var_plot_loop(param, cartopy_pmesh, 'viridis', units='m', climits=True, vmin=250, vmax=12500)
    # breakpoint()
    # #quiver_loop(param, cartopy_pmesh, 'viridis', units='K', climits=True)
    # param.var_label='flagged_diff_dthresh'

    # overlap_plot_loop( param, cartopy_pmesh, 'viridis',  'c1*-c1', units=' ',climits=True, vmin=-1, vmax=2)

    # param.var_label='flagged_warped_dthresh'

    # overlap_plot_loop( param, cartopy_pmesh, 'viridis','c2-c1*', units=' ',  climits=True, vmin=-1, vmax=2)

    # param.var_label='flagged_dthresh'
     
    # overlap_plot_loop( param, cartopy_pmesh, 'viridis','c2-c1',  units=' ', climits=True, vmin=-1, vmax=2)
    
    param.var_label='flagged_diff_d'+ param.var
     
    overlap_plot_loop( param, cartopy_pmesh, 'RdBu',  param.var+'1*-'+param.var+'1',units='m', climits=True, vmin=-1500, vmax=1500)
    
    param.var_label='flagged_d'+ param.var
     
    overlap_plot_loop( param, cartopy_pmesh, 'RdBu', param.var+'2-'+param.var+'1' , units='m',climits=True, vmin=-1500, vmax=1500)
    
    param.var_label='flagged_warped_d'+ param.var
     
    overlap_plot_loop( param, cartopy_pmesh, 'RdBu', param.var+'2-'+param.var+'1*',units='m',  climits=True, vmin=-1500, vmax=1500)
    

    # quiver_loop(param, cartopy_pmesh, 'viridis', units='K', climits=True)

    # #quiver_loop(param)

    # #prefix='OR_ABI-L1b-RadF-M6C14_G18'
    # #plot_loop(prefix, datelist, 'Rad', cartopy_pmesh, 'viridis', 'test_rad.png', units='', climits=True)

    
if __name__=='__main__':
    param=parameters()
    param.prefix='OR_ABI-L2-ACHA2KMF-M6_G18'
    param.var='HT'
    main(param)
