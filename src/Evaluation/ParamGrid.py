import numpy as np 
import pprint as pp
class ParamGrid:

    def __init__(self,param_limits,step=10):
        '''
        Parameters
        --------------------------------------
        params_extents : dict with pram name as key and [minValue, maxValue] as value
        ex : {'param1' : [0,12], ... }
        step :  int, number of division for each param 
        '''
        self.param_limits = param_limits
        self.step = step
    
    def getParams(self):
        ''' 
        return array of dict of params : 
        ex : [ {'param1' : 0,'param2' :0 }, {'param1' : 0,'param2' :1 }
               {'param1' : 1,'param2' :0 }, {'param1' : 0,'param2' :1 }]
        '''
        linspaces = [np.linspace(start,stop,self.step) 
                    for (start,stop) in self.param_limits.values()]
        coords = np.meshgrid(*linspaces)
        grid = np.dstack(coords)
        grid_pairs = grid.reshape(-1,2)
        self.params = [ dict(zip(self.param_limits.keys(), el)) for el in grid_pairs ]
        return self.params
