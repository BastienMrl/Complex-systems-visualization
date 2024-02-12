import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import math

from simulation import * 
class GOLSimulation(Simulation): 
    kernel = None
    def __init__(self, init_states = None, init_params : ParamList = None): 
        super().__init__(0, init_states, init_params)
        kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
        kernel += jnp.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])[:, :, jnp.newaxis, jnp.newaxis]
        kernel = jnp.transpose(kernel, [3, 2, 0, 1]) 

    def step(self) :
        state =  self.current_states[0]
        grid = state.grid
        out = lax.conv(grid, self.kernel, (1, 1), 'SAME')

        cdt_1 = out == 12 
        cdt_2 = out == 13
        cdt_3 = out == 3

        out = jnp.logical_or(cdt_1, cdt_2)
        out = jnp.logical_or(out, cdt_3)
        state.set_grid(out.astype(jnp.float32))
