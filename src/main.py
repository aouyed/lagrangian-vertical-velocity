#import calculators as calc
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
