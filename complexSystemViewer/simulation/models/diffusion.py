import jax.numpy as jnp
import jax.lax as lax
import jax.random
from jax import jit
from ..utils import Timer

from ..simulation import *

class DiffusionParameters(SimulationParameters):

    def __init__(self, id_prefix : str = "default"):
        super().__init__(id_prefix)

        #init
        self.is_random : bool
        self.grid_size : int
        self.nb_channels : int

        #rules
        self.kernel : jnp.ndarray
        self.decay : float

        #front

        self._init_param : list[Param] = [
            BoolParam(id_p= self._get_id_prefix() + "randomStart", name= self._get_name_prefix() + "Random start", default_value=True),
            IntParam(id_p= self._get_id_prefix() + "gridSize", name= self._get_name_prefix() + "Grid size",
                    default_value=100, min_value=0, step=1),
            IntParam(id_p= self._get_id_prefix() + "channels", name = self._get_name_prefix() + "Nb Channels", default_value=1,
                    min_value= 1, step=1)
        ]

        self._rules_param : list[Param] = [
            IntParam(id_p= self._get_id_prefix() + "kernel", name = self._get_name_prefix() + "Kernel Size",
                    default_value = 3, min_value = 1, max_value = 15, step=2),
            IntParam(id_p= self._get_id_prefix() + "diffusion", name = self._get_name_prefix() + "Diffusion",
                    default_value = 0.5, min_value = 0.1, max_value = 5, step = 0.05),
            FloatParam(id_p= self._get_id_prefix() + "decay", name = self._get_name_prefix() + "Decay", default_value = 0.1, min_value = 0.,
                    max_value = 1., step = 0.05)
        ]

        self.set_all_params()
    
    
    def rule_param_value_changed(self, idx: int, param: Param) -> None:
        match (idx):
            case 0 | 1:
                self._set_kernel()
            case 2:
                self.decay = param.value

    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match (idx):
            case 0:
                self.is_random = param.value
            case 1:
                self.grid_size = param.value
            case 2:
                self.nb_channels = param.value
        
    def _set_kernel(self):
        kernel_size : int = self._rules_param[0].value
        sigma : float = self._rules_param[1].value
        length = (kernel_size - 1) / 2

        ax = jnp.linspace(-length, length, kernel_size)
        gauss = jnp.exp(-0.5 * jnp.square(ax) / jnp.square(sigma))
        kernel = jnp.outer(gauss, gauss)
        self.kernel = kernel / jnp.sum(kernel)


    

class DiffusionSimulation(Simulation):
    

    def __init__(self, params : DiffusionParameters, needJSON : bool = True):
        super().__init__(needJSON=needJSON)
        self.initSimulation(params)

    def initSimulation(self, params : DiffusionParameters):
        self.params : DiffusionParameters = params

        self.current_states : GridState = None

        if self.params.is_random:
            self.init_random_sim()
        else:
            self.init_default_sim()

        def interaction(mask : jnp.ndarray, states : GridState):
            mask = jnp.expand_dims(mask, 2)

            states.grid = jnp.where(mask >= 0, mask, states.grid)
        
        self.interactions = [Interaction("0", interaction)]

        if (self.NEED_JSON):
            self.to_JSON_object()
        
            
    def init_default_sim(self):
        grid = jnp.zeros((self.params.grid_size, self.params.grid_size, 1))
        state = GridState(grid)
        self.current_states = state
        self.width = self.params.grid_size
        self.height = self.params.grid_size
        self.current_states.id = 0

    def init_random_sim(self):
        key = jax.random.PRNGKey(1701)
        grid = jax.random.uniform(key, (self.params.grid_size, self.params.grid_size, self.nb_channels), dtype=jnp.float64)
        state = GridState(grid)
        self.current_states = state
        self.width = self.params.grid_size
        self.height = self.params.grid_size
        self.current_states.id = 0        
            
    def _step(self):
        timer = Timer("Diffusion step")
        timer.start()
        state : GridState = self.current_states
        grid = state.grid

        out = apply_convolution(grid, self.kernel)
        out *= (1. - self.decay)
        state.set_grid(out)

        timer.stop()

    
@jit
def apply_convolution(grid : jnp.ndarray, kernel : jnp.ndarray):
    return jnp.dstack([jsp.signal.convolve2d(grid[:, :, c], kernel, mode = "same")
                            for c in range(grid.shape[-1])])