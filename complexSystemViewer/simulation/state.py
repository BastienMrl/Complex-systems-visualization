from abc import ABC, abstractmethod 
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .particle import *
import numpy as np
import time
class State(ABC): 
    width : int = None
    height : int = None
    id : int = 0
    c : int = 0

    def __init__(self, height : float, width : float):
        if height > 0 :
            self.height = height
        else :
            raise ValueError("Height must be a postive float")
        if width > 0 :
            self.width = width
        else :
            raise ValueError("width must be a postive float")
    
    @abstractmethod
    def to_JSON_object(self):
        pass


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

        domain = [self.c, self.id, self.width * self.height, self.grid.shape[2]]
        domain.append(float(np.min(x_row)))
        domain.append(float(np.max(x_row)))
        domain.append(float(np.min(y_row)))
        domain.append(float(np.max(y_row)))
        l = [x_row, y_row]

        for i in range(self.grid.shape[2]):
            val = self.grid[:, :, i].flatten().tolist()
            domain.append(float(0.))
            domain.append(float(1.))
            l.append(val)
        return [domain] + l

    def set_grid(self, grid) : 
        self.grid = grid

class ParticleState(State) :
    particles : jnp.ndarray = None
    nb_values : int = 1

    values_min : list[float] = []
    values_max : list[float] = []
    
   
    
    
    def __init__(self, width, height, values_min : list[float], values_max : list[float], particles : jnp.ndarray=None):
        
        if particles == None :
            self.particles = None
        else : 
            self.particles = particles
            self.nb_values = particles.shape[-1] - 2
            self.values_min = values_min
            self.values_max = values_max
        


        super().__init__(width, height)
      
    def to_JSON_object(self):
        x_row = self.particles[:, 0].tolist()
        y_row = self.particles[:, 1].tolist()

        domain = [self.c, self.id, len(self.particles), self.nb_values]
        domain.append(float(-self.width / 2))
        domain.append(float(self.width / 2))
        domain.append(float(-self.height / 2))
        domain.append(float(self.height / 2))

        l = [domain, x_row, y_row]

        for i in range(self.nb_values):
            l.append(self.particles[:, i + 2].tolist())
            domain.append(self.values_min[i])
            domain.append(self.values_max[i])

        return l
    
    def get_pos_x(self) -> jnp.array:
        return self.particles[:, 0]
    
    def get_pos_y(self) -> jnp.array:
        return self.particles[:, 1]
    
    def get_value(self, i : int) -> jnp.array:
        return self.particles[:, 2 + i]
        

class CombinaisonState(State):
    states : list[State] = None

    def __init__(self, states : list[State]):
        self.states = states

    def to_JSON_object(self):
        # ret = []
        # for value in self.states:
        #     ret.append(value)
        # return ret
        return self.states[0].to_JSON_object()