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
    grid = None
    grid_particle_class : int = None
    
    def __init__(self,grid, grid_particle_class = 0):
        
        self.grid = grid
        self.grid_particle_class = grid_particle_class
        s = jnp.shape(self.grid)
        w = int(s[2])
        h = int(s[3])


        super().__init__(w, h)
        self.update_particles()

    def to_JSON_object(self):
        grid2d = np.squeeze(self.grid)
        single_x_row = np.arange(0-(self.width-1)/2, (self.width-1)/2+1).tolist()
        single_y_row = np.arange(0-(self.height-1)/2, (self.height-1)/2+1).tolist()

        x_row = single_x_row * self.height
        y_row = [val for val in single_y_row for _ in range(self.width)]

        val_rown = grid2d.flatten()
        l = [x_row, y_row, val_rown.tolist()]
        return l

        

    def update_particles(self) :
        particles = list()
        grid2d = jnp.squeeze(self.grid)
        id_i = 0
        it = np.nditer(grid2d, order='C', flags=['multi_index'])
        for cell in it: 
            
            y,x= it.multi_index
            
            particles.append(Particle(id_i, x-(self.width-1)/2, y-(self.height-1)/2, float(cell), particle_class = self.grid_particle_class, is_aligned_grid =  True))
            id_i+=1
        self.particles =  particles

    def set_grid(self, grid) : 
        self.grid = grid
        self.update_particles()