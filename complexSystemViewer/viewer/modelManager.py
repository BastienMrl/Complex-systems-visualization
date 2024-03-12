from enum import Enum
import copy as cp
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation

class ModelEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"

class ModelManager(object):

    @staticmethod
    def get_simulation_model(model_name : str, states = None):
        match model_name :
            case "Gol":
                return GOLSimulation(init_states=states)
            case "Lenia":
                return LeniaSimulation(init_states=states)
            
    @staticmethod
    def get_default_rules(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.default_rules)
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.default_rules)

    
    @staticmethod
    def get_initialization_parameters(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.initialization_parameters)
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.initialization_parameters)

