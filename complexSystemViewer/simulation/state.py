from abc import ABC, abstractmethod 
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .particle import *
import numpy as np

class State(ABC): 
    width : int = None
    height : int = None

    particles : list = None

    def __init__(self, height : float, width : float, particles : list = None):
        if height > 0 :
            self.height = height
        else :
            raise ValueError("Height must be a postive float")
        if width > 0 :
            self.width = width
        else :
            raise ValueError("width must be a postive float")
        self.particles = particles
    
    def set_grid(self, grid) : 
        self.grid = grid
        


class GridState(State) :
    grid : jnp.ndarray = None
    grid_particle_class : int = None
    
    def __init__(self,grid, grid_particle_class = 0):
        
        self.grid = grid
        self.grid_particle_class = grid_particle_class
        s = jnp.shape(np.squeeze(self.grid))
        w = int(s[0])
        h = int(s[1])


        super().__init__(w, h)
        
    def to_JSON_object(self):
        single_x_row = np.arange(0-(self.width-1)/2, (self.width-1)/2+1).tolist()
        single_y_row = np.arange(0-(self.height-1)/2, (self.height-1)/2+1).tolist()

        x_row = single_x_row * self.height
        y_row = [val for val in single_y_row for _ in range(self.width)]

        domain = [self.width * self.height, self.grid.shape[2]]
        l = [domain, x_row, y_row]

        for i in range(self.grid.shape[2]):
            val = self.grid[:, :, i].flatten().tolist()
            l.append(val)

        return l

    def set_grid(self, grid) : 
        self.grid = grid