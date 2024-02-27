from abc import ABC, abstractmethod 
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .param import *
import numpy as np
import time
class Simulation(ABC): 
    
    s_id = None
    current_states = None
    parameters = None
    width = None
    height = None
     

    def __init__(self, id :int, init_states = None, init_params  = None): 
        s_id = None
        if init_states != None :
            self.current_states = init_states
        else : 
            raise ValueError("Initial State can't be None")
        
        self.width = init_states[0].width
        self.height = init_states[0].height
        for state in init_states : 
            if not jnp.isclose(state.width, self.width) or not jnp.isclose(state.height, self.height):
                raise ValueError("States of a simulation must be of same size")
        

        if init_params != None :
            self.parameter = init_params
        else :
            pass
            #raise ValueError("Initial parameters can't be None")
    
    @abstractmethod
    def step(self) : 
        pass

    def to_JSON_object(self) :
        t0 = time.time()
        """ particules = self.current_states[0].particles
        
        row_gen = (np.append(np.array([p.pos_x, p.pos_y], dtype=float),p.values) for p in particules)

        arr = np.fromiter(row_gen, object)
        
        tsl = np.transpose(np.stack(arr)).tolist() """
        tsl = self.current_states[0].to_JSON_object()
        print("json obj ok - ", 1000*(time.time()-t0), "ms\n")
        return tsl

    @abstractmethod
    def declare_params() :
        pass

    def getParams(): 
        pass #dict 
        
        
        
        
    