
import cv2
import numpy as np
import xarray as xr
import pandas as pd
import kinematics_tracker as kt
import config as config 
from centroidtracker import CentroidTracker
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle 
from scipy import stats
from celluloid import Camera


class cloudy_plot:
    """An object that tracks and tags clouds, and calculates kinematic 
    and thermodynamic properties `.
    Parameters
    ----------
    file : string, 
        Input file with calculated AMVs.
    
    Attributes
    ----------
    location_ : ndarray of shape (n_features,)
        Estimated robust location.
    
    """
    def __init__(self,clouds):
        self.clouds=clouds
        
  
    def quiver_hybrid(self, ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
        ds=ds.coarsen(lat=25, boundary='trim').mean().coarsen(lon=25, boundary='trim').mean()
        ax, fig, im=self.implot(ds, values, vmin, vmax, date,ax, fig, cmap, scatterv)
        #ds=ds.coarsen(lat=3, boundary='trim').mean().coarsen(lon=3, boundary='trim').mean()
        X,Y=np.meshgrid(ds['lon'].values,ds['lat'].values)
        Q = ax.quiver(X,Y, np.squeeze(ds['u'].values), np.squeeze(ds['v'].values))
        return   ax, fig, im
        
    def implot(self, ds, values, vmin, vmax, date,ax, fig, cmap, scatterv):
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(color='grey')
        im = ax.imshow(values, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, 
                       extent=[ds['lon'].min().item(),
                               ds['lon'].max().item(),ds['lat'].min().item(),ds['lat'].max().item()])
        return ax, fig, im
        
    def time_series_plotter(self,label, tag):
        ds=self.clouds.ds_time_series 
        ds_rolled=self.clouds.ds_time_series_rolled 

        ds_mean=self.clouds.ds_clouds_mean

        ds_mean=ds_mean.sel(lat=slice(0,25), lon=slice(-100, -75))
        ids=ds_mean['id_map'].values
        ids=ids[~np.isnan(ids)]
        ids, counts=np.unique(ids, return_counts=True)
        count_sort_ind = np.argsort(-counts)
        ids=ids[count_sort_ind]
  
        fig, ax= plt.subplots()
        
    
        #for idno in ds['id'].values:
        for idno in [ids[1]]:
            ds_unit=ds.sel(id=idno)
            dates=ds_unit['time'].values
            dates=pd.to_datetime(dates).hour
            ax.plot(dates,ds_unit[label].values, label=str(idno))
            ds_unit=ds_rolled.sel(id=idno)
            ax.plot(dates,ds_unit[label].values, label=str(idno)+'_rolled')

        ax.legend()
        #ax.set_ylim(0,150)
        ax.set_xlabel('hour')
        ax.set_ylabel('cloud top pressure')
    
        plt.savefig('../data/processed/plots/ts_'+label+tag+'.png', dpi=300)
        plt.show()
        plt.close()

    
    def plot_loop(self, ds, var, func, vmin, vmax, cmap,tag):
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
    
        animation = camera.animate()
        animation.save(config.PLOT_PATH+ var+'_'+tag+'.gif')
    
    def rand_cmap( self, nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
        """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
        :param nlabels: Number of labels (size of colormap)
        :param type: 'bright' for strong colors, 'soft' for pastel colors
        :param first_color_black: Option to use first color as black, True or False
        :param last_color_black: Option to use last color as black, True or False
        :param verbose: Prints the number of labels and shows the colormap. True or False
        :return: colormap for matplotlib
        """
        from matplotlib.colors import LinearSegmentedColormap
        import colorsys
        import numpy as np
    
    
        if type not in ('bright', 'soft'):
            print ('Please choose "bright" or "soft" for type')
            return
    
        if verbose:
            print('Number of labels: ' + str(nlabels))
    
        # Generate color map for bright colors, based on hsv
        if type == 'bright':
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.2, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]
    
            # Convert HSV list to RGB
            randRGBcolors = []
            for HSVcolor in randHSVcolors:
                randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))
    
            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]
    
            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
    
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    
        # Generate soft pastel colors, by limiting the RGB spectrum
        if type == 'soft':
            low = 0.6
            high = 0.95
            randRGBcolors = [(np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]
    
            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]
    
            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)
    
        # Display colorbar
        if verbose:
            from matplotlib import colors, colorbar
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))
    
            bounds = np.linspace(0, nlabels, nlabels + 1)
            norm = colors.BoundaryNorm(bounds, nlabels)
    
            cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                       boundaries=bounds, format='%1i', orientation=u'horizontal')

        return random_colormap
    
    def animate(self, tag):
        cmap = self.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)
        ds_total=self.clouds.ds_clouds_mean
        ds_total=ds_total.sel(lat=slice(0,25), lon=slice(-100, -75))
        ids=ds_total['id_map'].values
        ids=ids[~np.isnan(ids)]
        ids, counts=np.unique(ids, return_counts=True)
        count_sort_ind = np.argsort(-counts)
        ids=ids[count_sort_ind]
        ds_total=ds_total.where(ds_total.id_map==ids[1])
        
        self.plot_loop(ds_total, 'divergence_mean',self.quiver_hybrid, -10, 10,'RdBu',config.FOLDER+tag)    
        self.plot_loop(ds_total, 'pressure_vel_mean', self.quiver_hybrid, -0.1, 0.1,'RdBu',config.FOLDER+tag)
        self.plot_loop(ds_total, 'pressure_tendency_mean', self.quiver_hybrid, -0.1, 0.1,'RdBu',config.FOLDER+tag)
        self.plot_loop(ds_total, 'pressure_rate_mean', self.quiver_hybrid, -0.1, 0.1,'RdBu',config.FOLDER+ tag)
        self.plot_loop(ds_total, 'dp_morph', self.quiver_hybrid, -0.1, 0.1,'RdBu',config.FOLDER+ tag)
        self.plot_loop(ds_total, 'dp_morph_mean', self.quiver_hybrid, -0.1, 0.1,'RdBu',config.FOLDER+ tag)

        # cmap = c.rand_cmap(1000, type='bright', first_color_black=True, last_color_black=False, verbose=True)
        self.plot_loop(ds_total, 'cloud_top_pressure_mean', self.quiver_hybrid, 0, 1000,'viridis',config.FOLDER+tag)
        self.plot_loop(ds_total, 'id_map', self.implot, 0, 1000,cmap,config.FOLDER + tag)
       
        # m.plot_loop(ds_total, 'divergence_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
        # m.plot_loop(ds_total, 'vorticity_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
        # # m.plot_loop(ds_total, 'size_rate_mean', c.implot, -10, 10,'RdBu',m.FOLDER+tag)    
        # # m.plot_loop(ds_total, 'pressure_adv_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
        # m.plot_loop(ds_total, 'pressure_vel_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
        # m.plot_loop(ds_total, 'pressure_tendency_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+tag)
        # m.plot_loop(ds_total, 'pressure_rate_mean', c.implot, -0.1, 0.1,'RdBu',m.FOLDER+ tag)
        # # m.plot_loop(ds_total, 'thresh_map', c.implot, 0, 255,'viridis',m.FOLDER + tag)
        # # m.plot_loop(ds_total, 'size_map', c.implot, 0, 100,'viridis',m.FOLDER + tag)
        # m.plot_loop(ds_total, 'pressure_vel', c.implot, -2, 2,'RdBu',m.FOLDER + tag)
        # m.plot_loop(ds_total, 'pressure_tendency', c.implot, -2, 2,'RdBu',m.FOLDER + tag)
        # m.plot_loop(ds_total, 'cloud_top_pressure', c.implot, 0, 1000,'viridis',m.FOLDER + tag)
            
            
    
   