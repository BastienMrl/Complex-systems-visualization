""" class Particle(): 
    p_id : int = None
    pos_x : float = None
    pos_y : float = None
    values : tuple = None
    particle_class : int = None
    is_aligned_grid : bool = False

    def __init__(self, p_id, x,y,values : tuple):
        self.p_id = p_id
        self.pos_x = x
        self.pos_y = y
        self.values = values
        self.particle_class = particle_class
        

    def to_JSON_object(self) :
        return {
            "id" : self.p_id,
            "pos" : [self.pos_x,self.pos_y],
   
            "values" : self.values,
            "class" : self.particle_class
            #"is_aligned_grid" : self.is_aligned_grid

        } """ 

import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import jax.scipy as jsp
import numpy as np
import typing as t
from functools import partial
import math
from dataclasses import dataclass

@dataclass
class Particle:
    pos_x : float 
    pos_y : float 
    values : list()
    #values : jnp.ndarray =  field(init=False)


    def to_JSON_object():
        #TODO
        pass
    
    


    