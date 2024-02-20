from abc import ABC, abstractmethod 
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .param import *
import numpy as np
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
        arr = list()
        id_arr = np.array([])
        pos_arr_x = np.empty( shape=(0) )
        pos_arr_y= np.empty( shape=(0) )
        values_arr = np.empty( shape=(0) )
        class_arr = np.empty( shape=(0) )
        for state in self.current_states :
          
            particules = state.particles
            v_id = np.vectorize(lambda p: p.p_id)
            id_arr= v_id(particules)
            
            v_pos_x = np.vectorize(lambda p: p.pos_x)
            pos_arr_x = v_pos_x(particules)

            v_pos_y = np.vectorize(lambda p: p.pos_y)
            pos_arr_y = v_pos_y(particules)

            v_val = np.vectorize(lambda p: p.values)
            values_arr=v_val(particules)

            v_cla = np.vectorize(lambda p: p.particle_class)
            class_arr=v_cla(particules)
            
            
        arr.append(pos_arr_x.tolist())
        arr.append(pos_arr_y.tolist())
        arr.append(values_arr.tolist())
        return arr
        return {
            #'ids' : id_arr.tolist(),
            'x' : pos_arr_x.tolist(),
            'y' : pos_arr_y.tolist(),
            'values' : values_arr.tolist(),
            #'classes' : class_arr.tolist()  
        }
    