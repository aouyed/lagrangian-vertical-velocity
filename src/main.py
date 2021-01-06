
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as rgi
import numpy as np
import metpy.calc
from metpy.units import units
import cv2

def quiver_plotter(ds, title):

    ds=ds.coarsen(image_x=25).mean().coarsen(image_y=25).mean()
    fig, ax = plt.subplots()
    fig.tight_layout()
    Q = ax.quiver(ds['image_x'].values, ds['image_y'].values, ds['flow_x'].values, ds['flow_y'].values)
   
    ax.set_title('Observed Velocities')
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)
    print('plotted quiver...')




def drop_nan(frame):
    #row_mean = np.nanmean(frame, axis=1)
    #inds = np.where(np.isnan(frame))
    #frame[inds] = np.take(row_mean, inds[0])
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
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC)
    return res


def preprocessing():
    ds_m=xr.open_dataset('../data/GEOS.fp.fcst.inst3_3d_asm_Np.20201104_18+20201104_1800.V01.nc4')
    #ds_s=xr.open_dataset('../data/G16V04.0.ACTIV.2020309.1801.PX.02K.NC')
    #ds_s0=xr.open_dataset('../data/G16V04.0.ACTIV.2020309.1731.PX.02K.NC')
    ds_s=xr.open_dataset('../data/G16V04.0.ACTIV.2020335.1801.PX.02K.NC')
    ds_s0=xr.open_dataset('../data/G16V04.0.ACTIV.2020335.1731.PX.02K.NC')
    ds_s['cloud_top_height_0']=ds_s0['cloud_top_height'].copy()
    ds_s['image_y']=abs(ds_s['image_y']-800)
    
    ds_m_unit=ds_m.drop('time')
    frame=ds_s['cloud_top_height'].values
    frame0=ds_s['cloud_top_height_0'].values
    mask=np.isnan(frame)
    frame=np.nan_to_num(frame)
    frame0=np.nan_to_num(frame0)
    nframe0 = cv2.normalize(src=frame0, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    nframe = cv2.normalize(src=frame, dst=None,
                           alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    optical_flow = cv2.optflow.createOptFlow_DeepFlow()
    flowd = optical_flow.calc(nframe0, nframe, None)
    frame0d=warp_flow(frame0,flowd.copy())
    dz=frame-frame0d
    pz=frame-frame0
    dz[mask]=np.nan
    pz[mask]=np.nan
    dzdt=1000/1800*dz
    pzpt=1000/1800*pz
    
    ds_s['flow_x']=(('image_y','image_x'),flowd[:,:,0])
    ds_s['flow_y']=(('image_y','image_x'),flowd[:,:,1])
    ds_s['height_vel']=(('image_y','image_x'),dzdt)
    ds_s['height_tendency']=(('image_y','image_x'),pzpt)
    ds_s['height_tendency'].plot.hist(bins=100)
    ds_s['height_vel'].plot.hist(bins=100)
    plt.savefig('height_vel.png')
    plt.close()
    ds_s['height_vel'].plot.imshow(vmin=-2.5, vmax=2.5)
    plt.savefig('height_vel_map.png')
    plt.close()
    ds_s['height_tendency'].plot.imshow(vmin=-2.5, vmax=2.5)
    plt.savefig('height_tendency_map.png')
    plt.close()
    ds_s['cloud_top_height'].plot.imshow()
    plt.savefig('cloud_top_height.png')
    plt.close()
    ds_s['cloud_top_height_0'].plot.imshow()
    plt.savefig('cloud_top_height_0.png')
    plt.close()
    quiver_plotter(ds_s, 'quiver')
  
    breakpoint()
    
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
    
def main():
    preprocessing()
if __name__ == '__main__':
    main()
