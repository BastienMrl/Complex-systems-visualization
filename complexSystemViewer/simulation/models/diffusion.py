import jax.numpy as jnp
import jax.lax as lax
import jax.random
from ..utils import Timer

from ..simulation import *

class DiffusionSimulation(Simulation):
    initialization_parameters = [
        BoolParam(id_p="randomStart", name="Random start", default_value=True),
        IntParam(id_p="gridSize", name="Grid size",
                 default_value=100, min_value=0, step=1),
        IntParam(id_p="channels", name = "Nb Channels", default_value=1,
                 min_value= 1, step=1)
    ]

    default_rules = [
        IntParam(id_p="kernel", name = "Kernel Size",
                 default_value = 3, min_value = 1, max_value = 15, step=2),
        IntParam(id_p="diffusion", name = "Diffusion",
                 default_value = 0.5, min_value = 0.1, max_value = 5, step = 0.05),
        FloatParam(id_p="decay", name = "Decay", default_value = 0.1, min_value = 0.,
                   max_value = 1., step = 0.05)
    ]

    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True):
        super().__init__(needJSON=needJSON)
        self.initSimulation(rules, init_param)

    def initSimulation(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters):
        self.rules = rules
        self.init_param = init_param

        self.random_start = self.get_init_param("randomStart").value
        self.grid_size = self.get_init_param("gridSize").value
        self.nb_channels = self.get_init_param("channels").value
        self.kernel : jnp.array = None
        self.decay : float = 1.


        self._set_kernel()
        self._set_decay()
        self.current_states : GridState = None

        if self.random_start:
            self.init_random_sim()
        else:
            self.init_default_sim()

        if (self.NEED_JSON):
            self.to_JSON_object()
        
            
    def init_default_sim(self):
        grid = jnp.zeros((self.grid_size, self.grid_size, 1))
        state = GridState(grid)
        self.current_states = state
        self.width = self.grid_size
        self.height = self.grid_size
        self.current_states.id = 0

    def init_random_sim(self):
        key = jax.random.PRNGKey(1701)
        grid = jax.random.uniform(key, (self.grid_size, self.grid_size, self.nb_channels), dtype=jnp.float64)
        state = GridState(grid)
        self.current_states = state
        self.width = self.grid_size
        self.height = self.grid_size
        self.current_states.id = 0

    def updateRule(self, json):
        super().updateRule(json)
        self._set_decay()
        self._set_kernel()
        
            
    def _step(self):
        timer = Timer("Diffusion step")
        timer.start()
        state : GridState = self.current_states
        grid = state.grid

        out = jnp.dstack([jsp.signal.convolve2d(grid[:, :, c], self.kernel, mode = "same")
                            for c in range(grid.shape[-1])])
        out *= (1. - self.decay)
        state.set_grid(out)

        timer.stop()

    def _set_decay(self):
        decay : float = self.get_rules_param("decay").value
        self.decay = decay
    
    def _set_kernel(self):
        kernel_size : int = self.get_rules_param("kernel").value
        sigma : float = self.get_rules_param("diffusion").value
        length = (kernel_size - 1) / 2

        ax = jnp.linspace(-length, length, kernel_size)
        gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(sigma))
        kernel = jnp.outer(gauss, gauss)
        self.kernel = kernel / jnp.sum(kernel)