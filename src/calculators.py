
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
import main
import metpy
import pandas as pd
import matplotlib


GRID=0.018
R = 6371000

#LABELS=['entrainment','w','w*','w_s','pressure_vel','pressure_tendency',
 #           'p_error','adv']
LABELS=['pressure_vel','pressure_tendency',
           'p_error']
PLOT_PATH='../data/processed/plots/'
NC_PATH='../data/processed/netcdf/'
flow_var=main.flow_var
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
    ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
    ax, fig, im=implot(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv)
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

def warp_flow0(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow = flow.astype(np.float32)
    R2 = np.dstack(np.meshgrid(np.arange(w), np.arange(h)))
    pixel_map = R2 - 10*flow
    pixel_map=pixel_map.astype(np.float32)
    res = cv2.remap(img, pixel_map, None, cv2.INTER_LINEAR)
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



def map_plotter(ds, title, label, units_label='', vmin=0,vmax=0):
    values=np.squeeze(ds[label].values)
    print('frame shape')
    print(values.shape)
    fig, ax = plt.subplots()
    if vmin == vmax:
        im = ax.imshow(values, cmap='viridis', extent=[ds['lon'].min(
            ), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()])
    else:
           im = ax.imshow(values, cmap='RdBu', extent=[ds['lon'].min(
            ), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.title(label)
    plt.show()
    plt.close()  
    



def implot(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    im = ax.imshow(values, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, 
                   extent=[ds['lon'].min().item(),
                           ds['lon'].max().item(),ds['lat'].min().item(),ds['lat'].max().item()])
    return ax, fig, im

def implot_masked(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):

    #values= ds['cloud_top_pressure'].values
    mask=values>750
   # mask[mask==False]=np.nan
    print(mask)
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey')
    im = ax.imshow(values, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, 
                   extent=[ds['lon'].min().item(),
                           ds['lon'].max().item(),ds['lat'].min().item(),ds['lat'].max().item()])
    cmap1 = plt.get_cmap('tab20b')
    #cmap1.set_bad(alpha=0)
    ax.imshow(mask,cmap=cmap1)

    return ax, fig, im

def calc(ds_unit, frame0, pressure0, height0):
     frame=np.squeeze(ds_unit[flow_var].values)
     frame=np.nan_to_num(frame)
     height=np.squeeze(ds_unit['cloud_top_height'].values)
     mask=np.isnan(height)
     pressure=np.squeeze(ds_unit['cloud_top_pressure'].values)
     nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
     nframe = cv2.normalize(src=frame, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
     #need to test this 
     #optical_flow = cv2.optflow.createOptFlow_DeepFlow()
     #flowd = optical_flow.calc(frame0, frame, None)
     flowd=cv2.calcOpticalFlowFarneback(nframe0,nframe, None, 0.5, 3, 20, 3, 7, 1.2, 0)
     # ####
  
     height0=np.nan_to_num(height0)
     height0d=warp_flow(height0.copy(),flowd.copy())
     pressure0d=warp_flow(pressure0.copy(),flowd.copy())
     dz=height-height0d
     pz=height-height0
     dp=pressure-pressure0d
     pp=pressure-pressure0
     dz[mask]=np.nan
     dp[mask]=np.nan
     pp[mask]=np.nan
     pz[mask]=np.nan
     print('mean error:')
     print(np.nanmean(abs(pp)))
        
     dzdt=1000/1800*dz
     dpdt=1/1800*dp
     pppt=1/1800*pp
     pzpt=1000/1800*pz
     
     ds_unit['flow_x']=(('time','lat','lon'),np.expand_dims(flowd[:,:,0],axis=0))
     ds_unit['flow_y']=(('time','lat','lon'),np.expand_dims(flowd[:,:,1],axis=0))
     ds_unit=wind_calculator(ds_unit)
     ds_unit['height_tendency']=(('time','lat','lon'),np.expand_dims(pzpt,axis=0))
     ds_unit['height_vel']=(('time','lat','lon'),np.expand_dims(dzdt,axis=0))
     ds_unit['pressure_vel']=(('time','lat','lon'),np.expand_dims(dpdt,axis=0))
     ds_unit['pressure_tendency']=(('time','lat','lon'),np.expand_dims(pppt,axis=0))
     ds_unit['height0']=(('time','lat','lon'),np.expand_dims(height0,axis=0))
     ds_unit['height0d']=(('time','lat','lon'),np.expand_dims(height0d,axis=0))
     print('frame0')
     map_plotter(ds_unit, 'height0', 'height0')
     print('frame0d')
     map_plotter(ds_unit, 'height0d', 'height0d')
     map_plotter(ds_unit, 'cloud_top_height', 'cloud_top_height')
     map_plotter(ds_unit, 'cloud_top_pressure', 'cloud_top_pressure')
     frame0=frame
     pressure0=pressure
     height0=height
     nframe0=nframe
     print(abs(ds_unit['pressure_vel']).mean())
     print(abs(ds_unit['pressure_tendency']).mean())
     return ds_unit, frame0, height0, pressure0
 

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
    
def interpolation(ds_s,ds_m): 
    print('interpolating...')
    print(ds_m)
    t_function=rgi(points=(ds_m['level'].values, ds_m['latitude'].values, 
                           ds_m['longitude'].values),values= np.squeeze(ds_m['t'].values),
                   bounds_error=False, fill_value=np.nan)
    omega_function=rgi(points=(ds_m['level'].values, 
                               ds_m['latitude'].values, ds_m['longitude'].values),
                       values= np.squeeze(ds_m['w'].values),bounds_error=False, 
                       fill_value=np.nan)
    
    v_function=rgi(points=(ds_m['level'].values, ds_m['latitude'].values, 
                          ds_m['longitude'].values),
                  values= np.squeeze(ds_m['v'].values),
                  bounds_error=False, fill_value=np.nan)
    u_function=rgi(points=(ds_m['level'].values,
                           ds_m['latitude'].values, ds_m['longitude'].values),
                   values= np.squeeze(ds_m['u'].values),
                   bounds_error=False, fill_value=np.nan)
    df=ds_s[['cloud_top_pressure','cloud_top_height','height_vel','height_tendency','pressure_vel','pressure_tendency']].to_dataframe().reset_index()
    print(df['cloud_top_pressure'].max())
    df['omega']=omega_function(df[['cloud_top_pressure','lat', 'lon']].values)
    df['t']=t_function(df[['cloud_top_pressure','lat', 'lon']].values)
    df['u']=u_function(df[['cloud_top_pressure','lat', 'lon']].values)
    df['v']=v_function(df[['cloud_top_pressure','lat', 'lon']].values)
    df['surface_p']=1000
    df['surface_t']=t_function(df[['surface_p','lat', 'lon']].values)
    df['omega_s']=omega_function(df[['surface_p','lat', 'lon']].values)
    omega=df['omega'].to_numpy()*units('Pa/s')
    pressure=df['cloud_top_pressure'].to_numpy()*units('hPa')
    t=df['t'].to_numpy()*units('K')
    df['w*']=metpy.calc.vertical_velocity(omega, pressure, t )
    
    pressure=df['surface_p'].to_numpy()*units('hPa')
    t=df['surface_t'].to_numpy()*units('K')
    df['w_s']=metpy.calc.vertical_velocity(omega, pressure, t )
    df['w']=df['w*']-df['w_s']
    df=df.set_index(['lat', 'lon','time'])
    df=df.dropna()
    ds_inter=xr.Dataset.from_dataframe(df)
    print(ds_inter)
    return ds_inter

def omega_calculator(ds, label):
    df=ds.to_dataframe()
    df=df.reset_index()
    omega=df[label].to_numpy()*units('Pa/s')
    pressure=df['cloud_top_pressure'].to_numpy()*units('hPa')
    t=df['t'].to_numpy()*units('K')
    df[label]=metpy.calc.vertical_velocity(omega, pressure, t )
    df=df.dropna()
    df=df.set_index(['lat', 'lon','time'])
    ds=xr.Dataset.from_dataframe(df)
    return ds
    

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
    plt.ylim(yedges)
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
    
def marginal(ds, label, tag):
    ds[label].plot.hist(bins=100)
    plt.show()
    plt.savefig(main.PLOT_PATH + label +'_'+tag+'_marginal.png', dpi=300)
    plt.close()

def quiver_plot(ds, title):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['lon'].values, ds['lat'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds['u'].values), np.squeeze(ds['v'].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 1, r'$1$ m/s', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)

    plt.close()
    
    
def contourf_plotter(ds, title, label, units_label, vmin, vmax):
    values = np.squeeze(ds[label].values)
    values[values == 0] = np.nan
    fig, ax = plt.subplots()
    X, Y= np.meshgrid(ds['lon'].values,ds['lat'].values) 
    clevsf=np.linspace(vmin, vmax,10, endpoint=True)
    im = ax.contourf(X,Y,values, clevsf,cmap='viridis',extend="both", extent=[ds['lon'].min(), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.title(label)
    plt.tight_layout()
    plt.savefig('../data/processed/plots/countourf_'+title+'.png',
                bbox_inches='tight', dpi=300)
    
    plt.close()
    
    
def marginals(ds, tag):
    
    
    for label in LABELS:
        marginal(ds,label, tag)

    

def post_process(ds, tag):
    means={'var':[],'mean_of_mag':[]}
    
  
    
    for label in LABELS:
       means['var'].append(label)
       means['mean_of_mag'].append(abs(ds[label]).mean().item(0))

    df_results = pd.DataFrame(data=means)
    print(df_results)
    df_results.to_csv('../data/processed/csv/'+tag+'.csv')
    