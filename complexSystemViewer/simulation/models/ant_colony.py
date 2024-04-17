import jax.numpy as jnp
import jax.lax as lax
import jax.random
import jax.image as jimage

from complexSystemViewer.simulation.param import Param

from ..simulation import *
from ..models.diffusion import *
from ..models.game_of_life import *
from ..models.lenia import *
from .physarum_agent import *
from ..utils import Timer
import copy as cp

class AntColonyParameters(SimulationParameters):

    def __init__(self, agents : list[Simulation], diffusion : Simulation, id_prefix : str = "default"):
        super().__init__(id_prefix)

        self.agents

        #init
        self.grid_size_send : int
        self.grid_size : int
        
        # case grid_size

        #rules
        self.drop_amount : float

        #front
        self._init_param : list[Param] = [
            IntParam(id_p= self._get_id_prefix() + "gridsizeSend", name = self._get_name_prefix() + "Sended Grid Size",
                     default_value=150, min_value = 1, step= 1),
            IntParam(id_p= self._get_id_prefix() + "gridSize", name = self._get_name_prefix() + "Grid Size",
                     default_value = 150, min_value = 1, step= 1)
        ]

        self._rules_param : list[Param] = [
            FloatParam(id_p=self._get_id_prefix() + "dropAmount", name = self._get_name_prefix() + "Dropped Amount",
                       default_value=0.1, min_value = 0.0, stpe = 0.05)
        ]

        self.set_all_params()

    def rule_param_value_changed(self, idx: int, param: Param) -> None:
        match (idx):
            case 0:
                self.drop_amount = param.value
        
    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match(idx):
            case 0:
                self.grid_size_send = param.value
            case 1:
                self.grid_size = param.value
    




def substract_param_ids(initial : list[Param], to_substract : list[Param]):
    ret = initial
    for sub in to_substract:
        id = -1
        for i, param in enumerate(ret):
            if sub.id_param == param.id_param:
                id = i
        if (id != -1):
            ret.pop(id)
    return ret

def generate_init_parameters(own, shared, agent, diffusion):
    initialization_parameters = cp.deepcopy(own) + cp.deepcopy(shared)
    initialization_parameters += substract_param_ids(cp.deepcopy(agent), shared)
    initialization_parameters += substract_param_ids(cp.deepcopy(diffusion), shared)
    return initialization_parameters

def generate_rules_parameter(own, agent, diffusion):
    default_rules = own
    default_rules += agent
    default_rules += diffusion
    return default_rules

class AntColony(Simulation):
    

    # initialization parameters

    own_initialization_parameters = [
        IntParam(id_p="gridSizeSend", name = "Sended Grid size",
                 default_value=150, min_value = 1, step=1),
    ]

    shared_initialization_parameters = [
        IntParam(id_p="gridSize", name = "Grid size",
                 default_value=200, min_value = 1, step=1),
        IntParam(id_p="C", name="Number of Channels",
                    default_value=1, min_value=1, max_value=4, step=1)
    ]



    # rules parameters

    own_default_rules = [
        FloatParam(id_p="dropAmount", name = "Dropped amount", default_value = 0.1,
                   min_value = 0.0, step = 0.05)
    ]

    
    initialization_parameters = []
    default_rules = []

    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True,
                agent : Simulation = PhysarumAgentSimulation, diffusion : Simulation = DiffusionSimulation):
        super().__init__(needJSON=needJSON)
        self.agent_simulation : Simulation = agent
        self.diffusion_simulation : Simulation = diffusion
        self.initSimulation(rules, init_param)


    def initSimulation(self, rules : list[Param] = default_rules,
                        init_param : list[Param] = initialization_parameters):

        self.rules = rules
        self.init_param = init_param

        self.grid_size_send = self.get_init_param("gridSizeSend").value
        self.channels = self.get_init_param("C").value
        self.drop_amount = self.get_rules_param("dropAmount").value

        self.diffusion : Simulation  = self.diffusion_simulation(self._get_diffusion_rules(), self._get_diffusion_initialization_parameters(), False)
        self.agents : Simulation = self.agent_simulation(self._get_agents_rules(), self._get_agents_initialization_parameters(), False)

        self.current_states = GridState(jnp.zeros((self.grid_size_send, self.grid_size_send, 1)))


        self.width = self.diffusion.current_states.width
        self.height = self.diffusion.current_states.height


        def interaction(mask : jnp.ndarray, states : GridState):
            shape = self.diffusion.current_states.grid.shape
            
            self.diffusion.current_states.grid = jimage.resize(states.grid, shape, "linear")
            
            
            mask = jimage.resize(mask, (shape[0], shape[1]), "linear")

            self.diffusion.applyInteraction("0", mask)

            self.current_states.set_grid(jimage.resize(self.diffusion.current_states.grid, (self.grid_size_send, self.grid_size_send, 1), "linear"))
            
        
        self.interactions = [Interaction("0", interaction)]
        if (self.NEED_JSON):
            self.to_JSON_object()

    def updateRule(self, json):
        for param in self.rules:
            print(param.id_param)
        print(self.rules)
        super().updateRule(json)
        self.drop_amount = self.get_rules_param("dropAmount").value
        self.agents.updateRule(None)
        self.diffusion.updateRule(None)

    def _step(self):
        grid = self.diffusion.current_states.grid
        self.agents.set_grid(grid)
        self.agents.newStep()

        
        x = jnp.round(self.agents.current_states.get_pos_x()).astype(jnp.int16)
        y = jnp.round(self.agents.current_states.get_pos_y()).astype(jnp.int16)
        grid = grid.at[x, y, self.agents.channel].add(self.drop_amount)
        self.diffusion.current_states.set_grid(grid)
        self.diffusion.newStep()

        timer = Timer("Resizing grid")
        timer.start()
        self.current_states.set_grid(jimage.resize(self.diffusion.current_states.grid, (self.grid_size_send, self.grid_size_send, self.channels), "linear"))
        timer.stop()


    def _get_agents_initialization_parameters(self) -> list[Param]:
        ret = []
        for param in self.init_param:
            if (self._contains_param_id(self.shared_initialization_parameters + self.agent.initialization_parameters, param)):
                ret.append(param)
        return ret

    def _get_diffusion_initialization_parameters(self) -> list[Param]:
        ret = []
        for param in self.init_param:
            if (self._contains_param_id(self.shared_initialization_parameters + self.diffusion.initialization_parameters, param)):
                ret.append(param)
        return ret

    def _get_agents_rules(self) -> list[Param]:
        return self.agent.default_rules

    def _get_diffusion_rules(self) -> list[Param]:
        return self.diffusion.default_rules
    
    def _contains_param_id(self, l : list[Param], value : Param) -> bool:
        for param in l:
            if param.id_param == value.id_param:
                return True
        return False
    
class AntColonySimulation(AntColony):
    agent = PhysarumAgentSimulation
    diffusion = DiffusionSimulation


    initialization_parameters = generate_init_parameters(AntColony.own_initialization_parameters, AntColony.shared_initialization_parameters, 
                                                         agent.initialization_parameters, diffusion.initialization_parameters)

    default_rules = generate_rules_parameter(AntColony.own_default_rules, agent.default_rules, diffusion.default_rules)

    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True,
                 agent = agent, diffusion = diffusion):
        super().__init__(rules, init_param, needJSON, agent, diffusion)

    def initSimulation(self, rules : list[Param] = default_rules,
                        init_param : list[Param] = initialization_parameters):
        super().initSimulation(rules, init_param)
    
class PhysarumLeniaSimulation(AntColony):
    agent = PhysarumAgentSimulation
    diffusion = LeniaSimulation

    initialization_parameters = generate_init_parameters(AntColony.own_initialization_parameters, AntColony.shared_initialization_parameters, 
                                                         agent.initialization_parameters, diffusion.initialization_parameters)

    default_rules = generate_rules_parameter(AntColony.own_default_rules, agent.default_rules, diffusion.default_rules)

    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True,
                 agent = agent, diffusion = diffusion):
        super().__init__(rules, init_param, needJSON, agent, diffusion)

    def initSimulation(self, rules : list[Param] = default_rules,
                        init_param : list[Param] = initialization_parameters):
        super().initSimulation(rules, init_param)