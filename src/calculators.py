
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
import io
import cmocean
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import metpy.calc as mpcalc
from metpy.units import units



GRID=0.018
R = 6371000


PLOT_PATH='../data/processed/plots/'
NC_PATH='../data/processed/netcdf/'
flow_var='cloud_top_height'
DATE_FORMAT="%m/%d/%Y, %H:%M:%S"

def quiver_plotter(ds, title, date):
    
    date=date.strftime(DATE_FORMAT)
    ds=ds.coarsen(image_x=50, boundary='trim').mean().coarsen(image_y=50, boundary='trim').mean()
    fig, ax = plt.subplots()
    fig.tight_layout()
    Q = ax.quiver(ds['image_x'].values, ds['image_y'].values, ds['flow_x'].values, ds['flow_y'].values)
   
    ax.set_title('Observed Velocities')
    plt.savefig(PLOT_PATH+title+'_'+date+'.png', bbox_inches='tight', dpi=300)
    print('plotted quiver...')

def quiver_hybrid(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
    #ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
    ax, fig, im=implot(ds, values, vmin, vmax, date,ax, fig, cmap)
    #ds=ds.coarsen(lat=3, boundary='trim').mean().coarsen(lon=3, boundary='trim').mean()
    X,Y=np.meshgrid(ds['lon'].values,ds['lat'].values)
    Q = ax.quiver(X,Y, np.squeeze(ds['flow_x'].values), np.squeeze(ds['flow_y'].values))
    return   ax, fig, im

def scatter_hybrid(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
    ax, fig, im=implot(ds, values, vmin, vmax, date,ax, fig, cmap)
    
    ds=ds.coarsen(lat=100, boundary='trim').mean().coarsen(lon=100, boundary='trim').mean()
    df=ds[scatterv].to_dataframe().reset_index().dropna()
    df=df.loc[df[scatterv]>0]
    df[scatterv]=1
    X,Y=df['lon'],df['lat']

    
    #df.loc[df[scatterv]<0,scatterv]=-1
    #df.loc[df[scatterv]>0,scatterv]=1
    C=df[scatterv]
    Q=ax.scatter(X,Y,s=20,c=C, marker = 'o', cmap = 'gray');
    #Q = ax.quiver(X,Y, np.squeeze(ds['flow_x'].values), np.squeeze(ds['flow_y'].values))
    return   ax, fig, im
  



def drop_nan(frame):
    row_mean = np.nanmean(frame, axis=1)
    inds = np.where(np.isnan(frame))
    frame[inds] = np.take(row_mean, inds[0])
    frame=frame.astype(np.float32)
    mask = np.ma.masked_invalid(frame)
    mask = np.uint8(mask.mask)
    frame = np.nan_to_num(frame)
    frame = cv2.inpaint(frame, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
    print('inpainted')
    return frame

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    flow = flow.astype(np.float32)
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

    

def quick_plotter(ds_unit, date):
    date=date.strftime(DATE_FORMAT)
    

    ds_unit['height_vel'].plot.hist(bins=100) 
    plt.savefig(PLOT_PATH+'height_vel_'+date+'.png')
    plt.close()
    ds_unit['height_vel'].plot.imshow(vmin=-2.5, vmax=2.5)
    plt.savefig(PLOT_PATH+'height_vel_map_'+date+'.png')
    plt.close()
    ds_unit['height_tendency'].plot.imshow(vmin=-2.5, vmax=2.5)
    plt.savefig(PLOT_PATH+'height_tendency_map_'+date+'.png')
    plt.close()
    ds_unit['cloud_top_height'].plot.imshow(vmin=0,vmax=10)
    plt.savefig(PLOT_PATH+'cloud_top_height._'+date+'.png')
    plt.close()



def implot(ds, values, vmin, vmax, date,ax, fig, cmap):
    pmap = cmocean.cm.haline
    im = ax.imshow(values, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, 
                   extent=[ds['lon'].min().item(),
                           ds['lon'].max().item(),ds['lat'].min().item(),ds['lat'].max().item()])
    return ax, fig, im


def calc(ds_unit, frame0):
     frame=np.squeeze(ds_unit[flow_var].values)
     mask=np.isnan(frame)
     frame=np.nan_to_num(frame)
     nframe = cv2.normalize(src=frame, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
     optical_flow = cv2.optflow.createOptFlow_DeepFlow()
     flowd = optical_flow.calc(frame0, frame, None)
     frame0d=warp_flow(frame0,flowd.copy())
     dz=frame-frame0d
     pz=frame-frame0
     dz[mask]=np.nan
     pz[mask]=np.nan
     dzdt=1000/1800*dz
     pzpt=1000/1800*pz
     
     ds_unit['flow_x']=(('time','lat','lon'),np.expand_dims(flowd[:,:,0],axis=0))
     ds_unit['flow_y']=(('time','lat','lon'),np.expand_dims(flowd[:,:,1],axis=0))
     ds_unit=wind_calculator(ds_unit)
     ds_unit['height_vel']=(('time','lat','lon'),np.expand_dims(dzdt,axis=0))
     ds_unit['height_tendency']=(('time','lat','lon'),np.expand_dims(pzpt,axis=0))
     frame0=frame
     nframe0=nframe
     
     return ds_unit, frame0
 

def wind_calculator(ds):
    flow_x=np.squeeze(ds['flow_x'].values)
    flow_y=np.squeeze(ds['flow_y'].values)
    lon,lat=np.meshgrid(ds['lon'].values,ds['lat'].values)
    dthetax = GRID*flow_x
    dradsx = dthetax * np.pi / 180
    lat = lat*np.pi/180
    dx = R*abs(np.cos(lat))*dradsx
    u= dx/1800
    
    dthetay =GRID*flow_y
    dradsy = dthetay * np.pi / 180
    dy = R*dradsy
    v= dy/1800
    
    ds['u']=(('time','lat','lon'),np.expand_dims(u,axis=0))
    ds['v']=(('time','lat','lon'),np.expand_dims(v,axis=0))
    return ds


    return ds
    
def interpolation(): 
    v_function=rgi(points=(1000-ds_m_unit['lev'].values, ds_m_unit['lat'].values, ds_m_unit['lon'].values),values= np.squeeze(ds_m_unit['V'].values),bounds_error=False, fill_value=np.nan)
    u_function=rgi(points=(1000-ds_m_unit['lev'].values, ds_m_unit['lat'].values, ds_m_unit['lon'].values),values= np.squeeze(ds_m_unit['U'].values),bounds_error=False, fill_value=np.nan)
    t_function=rgi(points=(1000-ds_m_unit['lev'].values, ds_m_unit['lat'].values, ds_m_unit['lon'].values),values= np.squeeze(ds_m_unit['T'].values),bounds_error=False, fill_value=np.nan)
    omega_function=rgi(points=(1000-ds_m_unit['lev'].values, ds_m_unit['lat'].values, ds_m_unit['lon'].values),values= np.squeeze(ds_m_unit['OMEGA'].values),bounds_error=False, fill_value=np.nan)

    df=ds_s[['cloud_top_pressure','cloud_top_height','cloud_top_height_0']].to_dataframe().reset_index()
    df['cloud_top_pressure']=1000-df['cloud_top_pressure']
    df['cloud_top_height']=1000*df['cloud_top_height']
    df['cloud_top_height_0']=1000*df['cloud_top_height_0']
    print(df['cloud_top_pressure'].max())
    df['u']=u_function(df[['cloud_top_pressure','latitude', 'longitude']].values)
    df['v']=v_function(df[['cloud_top_pressure','latitude', 'longitude']].values)
    df['omega']=omega_function(df[['cloud_top_pressure','latitude', 'longitude']].values)
    df['t']=t_function(df[['cloud_top_pressure','latitude', 'longitude']].values)

    omega=df['omega'].to_numpy()*units('Pa/s')
    pressure=df['cloud_top_pressure'].to_numpy()*units('hPa')
    t=df['t'].to_numpy()*units('K')
    df['w']=metpy.calc.vertical_velocity(omega, pressure, t )

    df['cloud_top_pressure']=-df['cloud_top_pressure']+1000
    df=df.set_index(['image_x', 'image_y'])
    ds_inter=xr.Dataset.from_dataframe(df)
    return ds_inter

def scatter2d(ds, title, label, xedges, yedges):
    print('scattering...')
    fig, ax = plt.subplots()
    df=ds.to_dataframe().reset_index().dropna()
    df=df[label]
    X=df[label[0]]
    Y=df[label[1]]
    ax.scatter(X,Y, marker = 'o', facecolors='none', edgecolors='r')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.savefig(title+'_scatter2d.png', dpi=300)
    
def hist2d(ds, title, label, xedges, yedges):
    print('2dhistogramming..')

    bins = 100
    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    # labelds = label.copy()
    # labelds.append('cos_weight')
    df = ds.to_dataframe().reset_index().dropna()
    df = df[label]
    df = df.astype(np.float32)
    #df = df.loc[df[label[0]] != 0]
    
    img, x_edges, y_edges = np.histogram2d(df[label[0]].values, df[label[1]].values, bins=[
                               xbins, ybins])
    img=img.T
    #if 'speed' in label:
    #    breakpoint()
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, ax = plt.subplots()

    im = ax.imshow(img, origin='lower',
                   cmap='CMRmap_r', aspect='auto', extent=extent)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.tight_layout()

    plt.savefig(title+'_his2d.png', dpi=300)
    plt.close()

