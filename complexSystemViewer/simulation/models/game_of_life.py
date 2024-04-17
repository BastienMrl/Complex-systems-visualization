import jax.numpy as jnp
import jax.lax as lax
import jax.random


from ..simulation import * 

@dataclass
class GOLParameters(SimulationParameters):

    def __init__(self, id_prefix : str = "default"):
        super().__init__(id_prefix)

        # init
        self.is_random : bool
        self.grid_size : int

        # rules 
        self.birth_min : int
        self.birth_max : int

        self.survival_min : int
        self.survival_max : int

        # front
        self._rules_param : list[Param] = [
                RangeIntParam(id_p= self._get_id_prefix() + "birth", name= self._get_name_prefix() + "Birth",
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
                RangeIntParam(id_p= self._get_id_prefix() + "survival", name= self._get_name_prefix() + "Survival",
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
                                ))
                ]
        
        self._init_param : list[Param] = [
                BoolParam(id_p= self._get_id_prefix() + "random", name= self._get_name_prefix() + "Random Start",
                        default_value=False),
                IntParam(id_p =self._get_id_prefix() +  "gridSize", name = self._get_name_prefix() + "Grid Size",
                            default_value=20, min_value = 0, step =1)
            ]
        
        self.set_all_params()

    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match(idx):
            case 0:
                self.is_random = param.value
            case 1:
                self.grid_size = param.value

    def rule_param_value_changed(self, idx: int, param : Param) -> None:      
        match (idx):
            case 0 :
                self.birth_min = param.min_param
                self.birth_max = param.max_param
            case 1 :
                self.survival_min = param.min_param
                self.survival_max = param.max_param
            
    
        


class GOLSimulation(Simulation): 

    def __init__(self, params : GOLParameters = GOLParameters(), needJSON : bool = True): 
        super().__init__(needJSON=needJSON)
        self.initSimulation(params)

    def initSimulation(self, params : GOLParameters = GOLParameters()):

        self.param = params

        self.kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
        self.kernel += jnp.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])[:, :, jnp.newaxis, jnp.newaxis]
        self.kernel = jnp.transpose(self.kernel, [3, 2, 0, 1])

        if self.param.is_random:
            self.init_random_sim()
        else:
            self.init_default_sim() 
        self.to_JSON_object()

        def gol_interaction(mask : jnp.ndarray, states : GridState):
            mask = jnp.expand_dims(mask, (2))
            to_zero = jnp.logical_not(mask == 0)
            to_one = mask > 0
            states.grid = jnp.logical_or(states.grid, to_one).astype(jnp.float32)
            states.grid = jnp.logical_and(states.grid, to_zero).astype(jnp.float32)

        self.interactions : list[Interaction] = [Interaction("0", gol_interaction)]


    def _step(self) :
        state : GridState = self.current_states
        grid = state.grid
        grid = jnp.expand_dims(jnp.squeeze(grid), (0, 1))
        out = lax.conv(grid.astype(np.float64), self.kernel.astype(np.float64), (1, 1), 'SAME')

        cdt_1 = jnp.zeros_like(out)
        for i in range(self.param.birth_min, self.param.birth_max + 1):
            cdt_1 = jnp.logical_or(cdt_1, out == i)

        cdt_2 = jnp.zeros_like(out)
        for i in range(self.param.survival_min, self.param.survival_max + 1):
            cdt_1 = jnp.logical_or(cdt_1, out == 10+i)
        out = jnp.logical_or(cdt_1, cdt_2)
        out = jnp.expand_dims(jnp.squeeze(out), (2))

        state.set_grid(out.astype(jnp.float32))

    def init_default_sim(self):
        grid = jnp.zeros((self.param.grid_size, self.param.grid_size, 1))
        state = GridState(grid)
        self.current_states = state
        self.width = self.param.grid_size
        self.height = self.param.grid_size
        self.current_states.id = 0

    def init_random_sim(self):
        key = jax.random.PRNGKey(1701)
        grid = jax.random.uniform(key, (self.param.grid_size, self.param.grid_size, 1))
        grid = jnp.round(grid)
        state = GridState(grid)
        self.current_states = state
        self.width = self.param.grid_size
        self.height = self.param.grid_size
        self.current_states.id = 0

