
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
from pylab import MaxNLocator
from amv_calculator import amv_calculator
from cloudy_system import cloudy_system 
from cloudy_plot import cloudy_plot

    

def main():
    dt=1200    
    x=amv_calculator(dt)
    c=cloudy_system(x.ds_amv)
    plotter=cloudy_plot(c)
    plotter.animate('test')
if __name__ == '__main__':
    main()
