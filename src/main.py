#import calculators as calc
from amv_calculator import amv_calculator
from cloudy_system import cloudy_system 
from cloudy_plot import cloudy_plot
    

def main():
    dt=3600
    x=amv_calculator(dt)
    c=cloudy_system(x)
    plotter=cloudy_plot(c)
    plotter.animate('test')
    plotter.time_series_plotter('cloud_top_pressure',str(dt)+'temp')

if __name__ == '__main__':
    main()
