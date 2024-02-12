import jax.numpy as jnp
import jax.lax as lax
import jax.random
from Particle import *

class GridState(State) :
    grid : Array = None
    grid_particle_class : int = None
    
    def __init__(grid : Array, grid_particle_class = 0):
        
        self.grid = grid
        self.grid_particle_class = grid_particle_class
        s = jnp.shape(self.grid)
        w = float(s[0])
        h = float(s[1])


        super().__init__(w, h)
        self.update_particles()

    def update_particles(self, w = self.width, h = self.height) :
        particles = list(id, x,y,value,particle_class = 0, is_aligned_grid = false)
        
        id_i = 0
        for cell in self.grid.flatten(order='C') :
            x,y = jnp.unravel_index(id_i, jnp.shape(self.grid), order='C')
            particles.append(Particle(i, x-self.w/2, y-self.h/2), float(cell), particle_class = self.grid_particle_class, is_aligned_grid =  True)
        return particles

    def set_grid(self, grid) : 
        self.grid = grid
        self.update_particles()