import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import math

from ..simulation import * 

class GOLSimulation(Simulation): 

    initialization_parameters = [
        BoolParam(id_p="randomStart", name="Random start", default_value=False),
        IntParam(id_p="gridSize", name="Grid size",
                 default_value=20, min_value=0, step=1),
    ]

    default_rules = [
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

    def __init__(self, init_states = None, rules = default_rules): 
        super().__init__()
        self.initSimulation(init_states, rules)

    def initSimulation(self, init_states = None, rules = default_rules, init_param = initialization_parameters):

        self.random_start = [p for p in init_param if p.id_param == "randomStart"][0].value
        self.grid_size = [p for p in init_param if p.id_param == "gridSize"][0].value

        self.rules = rules
        self.kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
        self.kernel += jnp.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])[:, :, jnp.newaxis, jnp.newaxis]
        self.kernel = jnp.transpose(self.kernel, [3, 2, 0, 1])

        if init_states != None:
            self.current_states = init_states
            self.width = init_states[0].width
            self.height = init_states[0].height
        elif self.random_start:
            self.init_random_sim()
        else:
            self.init_default_sim() 

        self.interactions : list[Interaction] = [Interaction("toLife", golInteractionReplacement)]


    def step(self) :
        state =  self.current_states[0]
        grid = state.grid
        grid = jnp.expand_dims(jnp.squeeze(grid), (0, 1))
        out = lax.conv(grid.astype(np.float64), self.kernel.astype(np.float64), (1, 1), 'SAME')

        b_param : RangeIntParam = [p for p in self.rules if p.id_param == "birth"][0]
        s_param : RangeIntParam = [p for p in self.rules if p.id_param == "survival"][0]

        cdt_1 = jnp.zeros_like(out)
        for i in range(b_param.min_param.value, b_param.max_param.value + 1):
            cdt_1 = jnp.logical_or(cdt_1, out == i)

        cdt_2 = jnp.zeros_like(out)
        for i in range(s_param.min_param.value, s_param.max_param.value + 1):
            cdt_1 = jnp.logical_or(cdt_1, out == 10+i)
        out = jnp.logical_or(cdt_1, cdt_2)
        out = jnp.expand_dims(jnp.squeeze(out), (2))

        state.set_grid(out.astype(jnp.float32))

    def set_current_state_from_array(self, new_state):
        state = new_state[2]
        grid = jnp.asarray(state, dtype=jnp.float32).reshape(self.current_states[0].grid.shape)
        self.current_states[0].set_grid(grid)


    def init_default_sim(self):
        grid = jnp.zeros((self.grid_size, self.grid_size, 1))
        state = GridState(grid)
        self.current_states = [state]
        self.width = self.grid_size
        self.height = self.grid_size

    def init_random_sim(self):
        key = jax.random.PRNGKey(1701)
        grid = jax.random.uniform(key, (self.grid_size, self.grid_size, 1))
        grid = jnp.round(grid)
        state = GridState(grid)
        self.current_states = [state]
        self.width = self.grid_size
        self.height = self.grid_size


