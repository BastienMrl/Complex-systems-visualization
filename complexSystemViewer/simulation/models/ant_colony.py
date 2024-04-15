import jax.numpy as jnp
import jax.lax as lax
import jax.random
import jax.image as jimage

from ..simulation import *
from ..models.diffusion import *
from .physarum_agent import *
from ..utils import Timer

class AntColony(Simulation):
    # initialization parameters

    own_initialization_parameters = [
        IntParam(id_p="gridSizeSend", name = "Sended Grid size",
                 default_value=150, min_value = 1, step=1)
    ]

    shared_initialization_parameters = [
        IntParam(id_p="gridSize", name = "Grid size",
                 default_value=500, min_value = 1, step=1),
    ]

    agents_initialization_parameters = [
        IntParam(id_p="nbAgents", name="Nb of Agents", default_value= 10000,
                 min_value = 100, max_value= 50000, step=1000),
    ]

    diffusion_initialization_parameters = [
        BoolParam(id_p="randomStart", name="Random start", default_value=False),
        IntParam(id_p="channels", name = "Nb Channels", default_value=1,
                 min_value= 1, step=1)
    ]


    # rules parameters

    own_default_rules = [
        FloatParam(id_p="dropAmount", name = "Dropped amount", default_value = 0.5,
                   min_value = 0.05, step = 0.05)
    ]

    diffusion_default_rules = [
        IntParam(id_p="kernel", name = "Kernel Size",
                 default_value = 3, min_value = 1, max_value = 15, step=2),
        IntParam(id_p="diffusion", name = "Diffusion",
                 default_value = 0.5, min_value = 0.1, max_value = 5, step = 0.05),
        FloatParam(id_p="decay", name = "Decay", default_value = 0.2, min_value = 0.,
                   max_value = 1., step = 0.05)
    ]

    agents_default_rules = [
        FloatParam(id_p="speed", name="Speed", default_value=1.1,
                   min_value= 1., max_value= 20, step = 0.5),
        FloatParam(id_p="distSensor", name="Sensor distance", default_value=10,
                   min_value= 0.5, max_value=50, step= 0.1),
        FloatParam(id_p="angleSensor", name="Sensor angle", default_value=25,
                   min_value= 5, max_value=270, step=1),
        FloatParam(id_p="rotationAngle", name="Rotation", default_value=35,
                   min_value= 5, max_value=170, step=1)
    ]

    initialization_parameters = own_initialization_parameters + shared_initialization_parameters + agents_initialization_parameters + diffusion_initialization_parameters

    default_rules = own_default_rules + agents_default_rules + diffusion_default_rules
    

    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True):
        super().__init__(needJSON=needJSON)
        self.initSimulation(rules, init_param)


    def initSimulation(self, rules : list[Param] = default_rules,
                        init_param : list[Param] = initialization_parameters):
        self.rules = rules
        self.init_param = init_param

        self.grid_size_send = self.get_init_param("gridSizeSend").value
        self.drop_amount = self.get_rules_param("dropAmount").value

        self.diffusion = DiffusionSimulation(self._get_diffusion_rules(), self._get_diffusion_initialization_parameters(), False)
        self.agents = PhysarumAgentSimulation(self._get_agents_rules(), self._get_agents_initialization_parameters(), False)

        self.current_states = GridState(jnp.zeros((self.grid_size_send, self.grid_size_send, 1)))


        self.width = self.diffusion.current_states.width
        self.height = self.diffusion.current_states.height


        def interaction(mask : jnp.ndarray, states : CombinaisonState):
            mask = jnp.expand_dims(mask, (2))
            size = self.diffusion.current_states.width
            mask = jimage.resize(mask, (size, size, 1), "linear")

            self.diffusion.current_states.grid = jnp.where(mask >= 0, mask, 0.)
            self.current_states.set_grid(jimage.resize(self.diffusion.current_states.grid, (self.grid_size_send, self.grid_size_send, 1), "linear"))
            
        
        self.interactions = [Interaction("0", interaction)]
        if (self.NEED_JSON):
            self.to_JSON_object()

    def updateRule(self, json):
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
        grid = grid.at[x, y].add(self.drop_amount)
        self.diffusion.current_states.set_grid(grid)
        self.diffusion.newStep()

        self.current_states.set_grid(jimage.resize(self.diffusion.current_states.grid, (self.grid_size_send, self.grid_size_send, 1), "linear"))



    def get_rules() -> list[Param] | None:
        return AntColony.default_rules

    def get_initialization() -> list[Param] | None:
        return AntColony.initialization_parameters

    def _get_agents_initialization_parameters(self) -> list[Param]:
        ret = []
        for param in self.init_param:
            if (self._contains_param_id(self.shared_initialization_parameters + self.agents_initialization_parameters, param)):
                ret.append(param)
        return ret

    def _get_diffusion_initialization_parameters(self) -> list[Param]:
        ret = []
        for param in self.init_param:
            if (self._contains_param_id(self.shared_initialization_parameters + self.diffusion_initialization_parameters, param)):
                ret.append(param)
        return ret

    def _get_agents_rules(self) -> list[Param]:
        return self.agents_default_rules

    def _get_diffusion_rules(self) -> list[Param]:
        return self.diffusion_default_rules
    
    def _contains_param_id(self, l : list[Param], value : Param) -> bool:
        for param in l:
            if param.id_param == value.id_param:
                return True
        return False