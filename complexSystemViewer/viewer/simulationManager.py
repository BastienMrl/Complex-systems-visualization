from enum import Enum
import copy as cp
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation
from simulation.models.flocking import FlockingSimulation

class SimulationEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"
    FLOCKING = "Flocking"

class SimulationManager(object):

    @staticmethod
    def get_simulation_model(model_name : str, states = None):
        match model_name :
            case "Gol":
                return GOLSimulation(init_states=states)
            case "Lenia":
                return LeniaSimulation(init_states=states)
            case "Flocking":
                return FlockingSimulation(init_states=states)
            
    @staticmethod
    def get_default_rules(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.default_rules)
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.default_rules)
            case "Flocking":
                return cp.deepcopy(FlockingSimulation.default_rules)
    
    @staticmethod
    def get_initialization_parameters(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.initialization_parameters)
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.initialization_parameters)
            case "Flocking":
                return cp.deepcopy(FlockingSimulation.initialization_parameters)

