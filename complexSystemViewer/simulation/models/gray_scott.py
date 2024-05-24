import jax.numpy as jnp
import jax.lax as lax
import jax.random
from jax import jit

from ..utils import Timer


from ..simulation import * 

@dataclass
class GrayScottParameters(SimulationParameters):

    def __init__(self, id_prefix : str = "default"):
        super().__init__(id_prefix)

        # init
        self.grid_size : int
        self.is_random : bool

        # rules 
        # U + 2V -> 3V
        # V -> P


        self.dt : float

        self.Du : float
        self.Dv : float

        self.F : float
        self.k : float

        self.iter : int

        # front
        self._rules_param : list[Param] = [
                FloatParam(id_p= self._get_id_prefix() + "dt", name= self._get_name_prefix() + "dt",
                            default_value=1, min_value=0.01, max_value=0.5, step=0.01),
                FloatParam(id_p=self._get_id_prefix() + "diffusion", name=self._get_name_prefix() + "Diffusion",
                           default_value = 0.5, min_value= 0.05, max_value=4, step=0.01),
                # FloatParam(id_p=self._get_id_prefix() + "Du", name=self._get_name_prefix() + "Du",
                #            default_value=0.5, min_value=0., max_value=1., step=0.01),
                # FloatParam(id_p=self._get_id_prefix() + "Dv", name=self._get_name_prefix() + "Dv",
                #            default_value=0.2, min_value=0., max_value=1., step=0.01),
                FloatParam(id_p=self._get_id_prefix() + "F", name=self._get_name_prefix() + "F",
                           default_value=0.037, min_value=0., max_value=0.1, step=0.001),
                FloatParam(id_p=self._get_id_prefix() + "k", name=self._get_name_prefix() + "k",
                           default_value=0.060, min_value=0., max_value=0.1, step=0.001),
                # IntParam(id_p=self._get_id_prefix() + "iter", name=self._get_name_prefix() + "Iterations per step",
                #          default_value=1, min_value=1, max_value=50)
                ]
        
        self._init_param : list[Param] = [
                IntParam(id_p =self._get_id_prefix() +  "gridSize", name = self._get_name_prefix() + "Grid Size",
                            default_value=100, min_value = 0, step =1),
                BoolParam(id_p=self._get_id_prefix() + "Random", name = self._get_name_prefix() + "Random",
                          default_value=False)
            ]
        
        self.set_all_params()

    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match(idx):
            case 0:
                self.grid_size = param.value
            case 1 :
                self.is_random = param.value

    def rule_param_value_changed(self, idx: int, param : Param) -> None:      
        match (idx):
            case 0 :
                self.dt = param.value
            case 1 : 
                self.Du = param.value
                self.Dv = param.value / 2.
            case 2 : 
                self.F = param.value
            case 3 : 
                self.k = param.value
            case 4 :
                self.iter = param.value


class GrayScottInteractions(SimulationInteractions):
    def __init__(self):
        super().__init__()

        self.interactions : dict[str, Callable[[jnp.ndarray, Simulation]]] = {
            "Add U" : partial(self._set_channel_value_with_mask, 0),
            "Add V" : partial(self._set_channel_value_with_mask, 1),

        }

laplacian_kernel = jnp.array([
                    [0.05, 0.2, 0.05],
                    [0.2, -1., 0.2],
                    [0.05, 0.2, 0.05]
                ])
@jit                
def laplacian(A : jnp.ndarray):
    """
    A : (x, y, c)
    ret : (x, y, c)
    """
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], laplacian_kernel, mode = 'same') 
                    for c in range(A.shape[-1])])

        


class GrayScottSimulation(Simulation): 

    # Channel 0 : U
    # Channel 1 : V

    def __init__(self, params : GrayScottParameters = GrayScottParameters(), needJSON : bool = True): 
        super().__init__(params, needJSON=needJSON)
        self.init_simulation(params)

    def init_simulation(self, params : GrayScottParameters = GrayScottParameters()):

        self.params : GrayScottParameters = params

        self.current_states : GridState

        if self.params.is_random:
            self.init_random_sim()
        else:
            self.init_default_sim() 
        self.to_JSON_object()


        self.interactions = GrayScottInteractions()


    def _step(self) :
        grid = self.current_states.grid
        grid = iterations(grid, self.params.dt, self.params.Du, self.params.Dv, self.params.F, self.params.k)
        self.current_states.set_grid(grid)


    def init_default_sim(self):
        SX = SY = self.params.grid_size
        mx, my = SX//2, SY//2 # center coordinated
        offsetX, offsetY= round(SX/8), round(SY/8)

        ones = jnp.ones((self.params.grid_size, self.params.grid_size, 1))
        zeros = jnp.zeros((SX, SY, 1)).at[mx-offsetX:mx+offsetX, my-offsetY:my+offsetY, :].set(
            jnp.ones((2*offsetX, 2*offsetY, 1))
        )
        grid = jnp.dstack((ones, zeros))
        state = GridState(grid)
        self.current_states = state
        self.width = self.params.grid_size
        self.height = self.params.grid_size
        self.current_states.id = 0

    def init_random_sim(self):
        key = jax.random.PRNGKey(1701)
        grid = jax.random.uniform(key, (self.params.grid_size, self.params.grid_size, 2))
        grid = jnp.round(grid)
        state = GridState(grid)
        self.current_states = state
        self.width = self.params.grid_size
        self.height = self.params.grid_size
        self.current_states.id = 0

@jit
def iterations(grid : jnp.ndarray, dt : float, Du : float, Dv : float, F : float, k : float):

    for _ in range(30):
    
        L = laplacian(grid)

        diffusion = jnp.array([[[Du, Dv]]]) * L

        uv_2 = jnp.expand_dims(grid[:, :, 0] * grid[:, :, 1] * grid[:, :, 1], 2)

        delta = diffusion + jnp.array([[[-1, 1]]]) * uv_2

        delta = delta + jnp.array([[[- F, - F - k]]]) * grid + jnp.array([[[F, 0]]])

        grid = grid + delta * dt

    return grid