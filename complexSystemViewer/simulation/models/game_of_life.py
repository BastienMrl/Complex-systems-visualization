import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import math

from ..simulation import * 

class GOLSimulation(Simulation): 

    default_parameters = [
        IntParam(id_p="gridSize", name="Grid size",
                 default_value=50, min_value=0, step=1),
        RangeIntParam(id_p="birth", name="Birth",
                      min_param= IntParam(
                          id_p="",
                          name="",
                          default_value=3,
                          min_value=0,
                          max_value=8,
                          step=1
                      ),
                      max_param= IntParam(
                          id_p="",
                          name="",
                          default_value=3,
                          min_value=0,
                          max_value=8,
                          step=1
                      )),
        RangeIntParam(id_p="survival", name="Survival",
                      min_param= IntParam(
                          id_p="",
                          name="",
                          default_value=2,
                          min_value=0,
                          max_value=8,
                          step=1
                      ),
                      max_param= IntParam(
                          id_p="",
                          name="",
                          default_value=3,
                          min_value=0,
                          max_value=8,
                          step=1
                      ))]

    def __init__(self, init_states = None, init_params = default_parameters): 
        super().__init__(0, init_states, init_params)
        self.name = "Game of life"
        self.kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
        self.kernel += jnp.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])[:, :, jnp.newaxis, jnp.newaxis]
        self.kernel = jnp.transpose(self.kernel, [3, 2, 0, 1]) 

    def step(self) :
        state =  self.current_states[0]
        grid = state.grid
        out = lax.conv(grid, self.kernel, (1, 1), 'SAME')

        b_param : RangeIntParam = [p for p in self.parameters if p.id_param == "birth"][0]
        s_param : RangeIntParam = [p for p in self.parameters if p.id_param == "survival"][0]

        cdt_1 = jnp.zeros_like(out)
        for i in range(b_param.min_param.value, b_param.max_param.value + 1):
            cdt_1 = jnp.logical_or(cdt_1, out == i)

        cdt_2 = jnp.zeros_like(out)
        for i in range(s_param.min_param.value, s_param.max_param.value + 1):
            cdt_1 = jnp.logical_or(cdt_1, out == 10+i)
        out = jnp.logical_or(cdt_1, cdt_2)

        #10 = cell alive 
        # cdt_1 = out == 12 #stay alive if 2 n
        # cdt_2 = out == 13
        # cdt_3 = out == 3    #spawn if 3 n
        # out = jnp.logical_or(cdt_1, cdt_2)
        # out = jnp.logical_or(out, cdt_3)


        state.set_grid(out.astype(jnp.float32))

    def set_current_state_from_array(self, new_state):
        state = new_state[2]
        width = self.current_states[0].width
        height = self.current_states[0].height
        grid = jnp.asarray(state, dtype=jnp.float32).reshape((width, height))
        grid = jnp.expand_dims(grid, (0, 1))
        self.current_states[0].set_grid(grid)
