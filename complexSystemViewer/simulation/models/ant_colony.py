import jax.numpy as jnp
import jax.lax as lax
import jax.random

from ..simulation import *
from ..models.diffusion import *
from .physarum_agent import *
from ..utils import Timer

class AntColony(Simulation):

    initialization_parameters = [
        IntParam(id_p="nbAgents", name="Nb of Agents", default_value= 500,
                 min_value = 1, max_value= 10000, step=1),
        IntParam(id_p="gridSize", name = "Grid Size",
                 default_value=100, min_value = 10, step=10)
    ]

    default_rules = [
        FloatParam(id_p="speed", name="Speed", default_value=1.,
                   min_value= 1., max_value= 20, step = 0.5),
        FloatParam(id_p="distSensor", name="Sensor distance", default_value=1.,
                   min_value= 0.5, max_value=5, step= 0.1),
        FloatParam(id_p="angleSensor", name="Sensor angle", default_value=45,
                   min_value= 10, max_value=270, step=1),
        FloatParam(id_p="rotationAngle", name="Rotation", default_value=90,
                   min_value=10, max_value=170, step=1),

        IntParam(id_p="kernel", name = "Kernel Size",
                 default_value = 3, min_value = 1, max_value = 15, step=2),
        IntParam(id_p="diffusion", name = "Diffusion",
                 default_value = 0.15, min_value = 0.1, max_value = 5, step = 0.05),
        FloatParam(id_p="decay", name = "Decay", default_value = 0.05, min_value = 0.,
                   max_value = 1., step = 0.05),
        FloatParam(id_p="dropAmount", name = "Dropped amount", default_value = 0.5,
                   min_value = 0.05, step = 0.05)
    ]

    diffusion_initialization_parameters = [
        BoolParam(id_p="randomStart", name="Random start", default_value=False),
        IntParam(id_p="gridSize", name="Grid size",
                 default_value=100, min_value=0, step=1),
        IntParam(id_p="channels", name = "Nb Channels", default_value=1,
                 min_value= 1, step=1)
    ]

    diffusion_default_rules = [
        IntParam(id_p="kernel", name = "Kernel Size",
                 default_value = 3, min_value = 1, max_value = 15, step=2),
        IntParam(id_p="diffusion", name = "Diffusion",
                 default_value = 0.5, min_value = 0.1, max_value = 5, step = 0.05),
        FloatParam(id_p="decay", name = "Decay", default_value = 0.2, min_value = 0.,
                   max_value = 1., step = 0.05)
    ]

    agents_initialization_parameters = [
        IntParam(id_p="nbAgents", name="Nb of Agents", default_value= 10000,
                 min_value = 1, max_value= 10000, step=1),
        IntParam(id_p="gridSize", name = "Grid Size",
                 default_value=100, min_value = 10, step=10)
    ]

    agents_default_rules = [
        FloatParam(id_p="speed", name="Speed", default_value=1.1,
                   min_value= 1., max_value= 20, step = 0.5),
        FloatParam(id_p="distSensor", name="Sensor distance", default_value=10,
                   min_value= 0.5, max_value=5, step= 0.1),
        FloatParam(id_p="angleSensor", name="Sensor angle", default_value=25,
                   min_value= 10, max_value=270, step=1),
        FloatParam(id_p="rotationAngle", name="Rotation", default_value=35,
                   min_value=10, max_value=170, step=1)
    ]

    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True):
        super().__init__(needJSON=needJSON)
        self.initSimulation(rules, init_param)


    def initSimulation(self, rules : list[Param] = default_rules,
                        init_param : list[Param] = initialization_parameters):
        self.rules = rules
        self.init_param = init_param

        self.drop_amount = self.get_rules_param("dropAmount").value

        self.diffusion = DiffusionSimulation(self.diffusion_default_rules, self.diffusion_initialization_parameters, False)
        self.agents = PhysarumAgentSimulation(self.agents_default_rules, self.agents_initialization_parameters, False)

        self.current_states = CombinaisonState([self.diffusion.current_states, self.agents.current_states])

        self.width = self.diffusion.current_states.width
        self.height = self.diffusion.current_states.height


        def interaction(mask : jnp.ndarray, states : CombinaisonState):
            mask = jnp.expand_dims(mask, (2))

            states.states[0].grid = jnp.where(mask >= 0, mask, 0.)
            self.diffusion.current_states = states.states[0]
            self.agents.current_states = states.states[1]
            
        
        self.interactions = [Interaction("0", interaction)]

    def updateRule(self, json):
        super().updateRule(json)
        self.drop_amount = self.get_rules_param("dropAmount").value

    def _step(self):
        grid = self.diffusion.current_states.grid
        self.agents.set_grid(grid)
        self.agents.newStep()

        
        x = jnp.round(self.agents.current_states.get_pos_x()).astype(jnp.int16)
        y = jnp.round(self.agents.current_states.get_pos_y()).astype(jnp.int16)
        grid = grid.at[x, y].add(self.drop_amount)
        self.diffusion.current_states.set_grid(grid)


        self.diffusion.newStep()