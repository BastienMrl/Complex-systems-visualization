import jax.numpy as jnp
import jax.lax as lax
import jax.image as jimage

from ..simulation import *
from ..models.diffusion import *
from ..models.game_of_life import *
from ..models.lenia import *
from .physarum_agent import *
from ..utils import Timer
import copy as cp

@dataclass
class _SimulationPair():
    params : SimulationParameters
    type : type

class AntColonyParameters(SimulationParameters):

    def __init__(self, agents : list[_SimulationPair], diffusion : _SimulationPair, id_prefix : str = "default"):
        super().__init__(id_prefix)

        self.agents : list[_SimulationPair] = agents
        self.diffusion : _SimulationPair = diffusion


        #init
        self.grid_size_send : int
        self.grid_size : int
        
        # case grid_size

        #rules
        self.drop_amount : list[float] = [None] * len(self.agents)

        #front
        self._init_param : list[Param] = [
            IntParam(id_p= self._get_id_prefix() + "gridsizeSend", name = self._get_name_prefix() + "Sended Grid Size",
                     default_value=150, min_value = 1, step= 1),
            IntParam(id_p= self._get_id_prefix() + "gridSize", name = self._get_name_prefix() + "Grid Size",
                     default_value = 150, min_value = 1, step= 1)
        ]

        self._rules_param : list[Param] = [
            FloatParam(id_p=self._get_id_prefix() + f"dropAmount{i}", name = self._get_name_prefix() + f"Dropped Amount {i}",
                       default_value=0.1, min_value = 0.0, step = 0.05) for i in range(len(self.drop_amount))
        ]

        self.set_all_params()

    def get_init_parameters(self) -> list[Param]:
        ret = self._init_param.copy()
        for agent in self.agents:
            ret += agent.params.get_init_parameters()
        ret += self.diffusion.params.get_init_parameters()
        return [param for param in ret if ((not "gridSize" in param.id_param) or (self._get_id_prefix() in param.id_param))]

    def get_rules_paramaters(self) -> list[Param]:
        ret = self._rules_param.copy()
        for agent in self.agents:
            ret += agent.params.get_rules_paramaters()
        return ret + self.diffusion.params.get_rules_paramaters()

    def update_rules_param(self, json: dict[str, str | float | int]):
        super().update_rules_param(json)
        for agent in self.agents:
            agent.params.update_rules_param(json)
        self.diffusion.params.update_rules_param(json)

    def update_init_param(self, json: dict[str, str | float | int]):
        super().update_init_param(json)
        for agent in self.agents:
            agent.params.update_init_param(json)
        self.diffusion.params.update_init_param(json)

    def set_init_from_list(self, source: list[Param]):
        for agent in self.agents:
            agent.params.set_init_from_list(source)
        self.diffusion.params.set_init_from_list(source)
        super().set_init_from_list(source)

    def rule_param_value_changed(self, idx: int, param: Param) -> None:
        if (not param.id_param.startswith(self.id_prefix)):
            return
        self.drop_amount[idx] = param.value
        
    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match(idx):
            case 0:
                self.grid_size_send = param.value
            case 1:
                self.grid_size = param.value
                self.diffusion.params.grid_size = self.grid_size
                for agent in self.agents:
                    agent.params.grid_size = self.grid_size
    

class AntColonyInteractions(SimulationInteractions):
    def __init__(self):
        super().__init__()

        def interaction(diffusion_id : int, mask : jnp.ndarray, simulation : AntColonySimulation):
            shape = simulation.diffusion.current_states.grid.shape
            
            simulation.diffusion.current_states.grid = jimage.resize(simulation.current_states.grid, shape, "linear")
            
            
            mask = jimage.resize(mask, (shape[0], shape[1]), "linear")

            simulation.diffusion.apply_interaction(diffusion_id, mask)

            sended_shape = simulation.current_states.grid.shape

            simulation.current_states.set_grid(jimage.resize(simulation.diffusion.current_states.grid, (sended_shape), "linear"))

        self.interactions : dict[str, Callable[[jnp.ndarray, AntColonySimulation]]] = {
            "Channel 1" : partial(interaction, "Channel 1"),
            "Channel 2" : partial(interaction, "Channel 2"),
            "Channel 3" : partial(interaction, "Channel 3"),
            "Channel 4" : partial(interaction, "Channel 4"),
            "Channel 5" : partial(interaction, "Channel 5"),
        }

class AntColony(Simulation):
    

    def __init__(self, params : AntColonyParameters, needJSON : bool = True):
        super().__init__(params, needJSON=needJSON)
        self.init_simulation(params)


    def init_simulation(self, params : AntColonyParameters):

        self.params : AntColonyParameters = params

        self.diffusion : Simulation  = self.params.diffusion.type(self.params.diffusion.params, False)
        self.agents : list[Simulation] = [agent.type(agent.params, False) for agent in self.params.agents]

        self.current_states = GridState(jnp.zeros((self.params.grid_size_send, self.params.grid_size_send, self.diffusion.current_states.grid.shape[-1])))


        self.width = self.diffusion.current_states.width
        self.height = self.diffusion.current_states.height
           
        
        self.interactions = AntColonyInteractions()

        if (self.NEED_JSON):
            self.to_JSON_object()


    def _step(self):
        grid = self.diffusion.current_states.grid
        new_grid = jnp.copy(grid)

        for i, agent in enumerate(self.agents):
            agent.set_grid(grid)
            agent.new_step()
            x = jnp.round(agent.current_states.get_pos_x()).astype(jnp.int16)
            y = jnp.round(agent.current_states.get_pos_y()).astype(jnp.int16)
            new_grid = new_grid.at[x, y, agent.params.sensing_channel].add(self.params.drop_amount[i])
               
        self.diffusion.current_states.set_grid(new_grid)
        self.diffusion.new_step()

        timer = Timer("Resizing grid")
        timer.start()
        self.current_states.set_grid(jimage.resize(self.diffusion.current_states.grid, (self.params.grid_size_send, self.params.grid_size_send, self.diffusion.current_states.grid.shape[-1]), "linear"))
        timer.stop()

    
class AntColonySimulation(AntColony):
    agent = _SimulationPair(PhysarumAgentParameters("Physarum"), PhysarumAgentSimulation)
    diffusion = _SimulationPair(DiffusionParameters("Diffusion"), DiffusionSimulation)

    params = AntColonyParameters([agent], diffusion)
    

    def __init__(self, params : AntColonyParameters = params, needJSON : bool = True):
        super().__init__(params, needJSON)
    
class PhysarumLeniaSimulation(AntColony):
    agent = _SimulationPair(PhysarumAgentParameters("Physarum"), PhysarumAgentSimulation)
    diffusion = _SimulationPair(LeniaParameters("Lenia"), LeniaSimulation)

    params = AntColonyParameters([agent], diffusion)

    def __init__(self, params : AntColonyParameters = params, needJSON : bool = True):
        super().__init__(params, needJSON)

class PhysarumLeniaBisSimulation(AntColony):
    agent_1 = _SimulationPair(PhysarumAgentParameters("1"), PhysarumAgentSimulation)
    agent_2 = _SimulationPair(PhysarumAgentParameters("2"), PhysarumAgentSimulation)
    diffusion = _SimulationPair(LeniaParameters("Lenia"), LeniaSimulation)

    params = AntColonyParameters([agent_1, agent_2], diffusion)

    def __init__(self, params : AntColonyParameters = params, needJSON : bool = True):
        super().__init__(params, needJSON)
