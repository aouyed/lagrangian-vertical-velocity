from amv_calculator import amv_calculator
from cloudy_system import cloudy_system 
from cloudy_plot import cloudy_plot
    

def main():
    dt=3600
    x=amv_calculator(dt)
    c=cloudy_system(x)
    plotter=cloudy_plot(c)
    plotter.time_series_plotter('cloud_top_pressure',str(dt)+'temp')
    plotter.time_series_plotter('pressure_rate',str(dt)+'temp')
    plotter.time_series_plotter('pressure_tendency',str(dt)+'temp')
    plotter.time_series_plotter('pressure_vel',str(dt)+'temp')
    plotter.time_series_plotter('dp_morph',str(dt)+'temp')
    plotter.time_series_plotter('pp_morph',str(dt)+'temp')
    plotter.time_series_plotter('size_rate',str(dt)+'temp')



    plotter.animate('test')

if __name__ == '__main__':
    main()
