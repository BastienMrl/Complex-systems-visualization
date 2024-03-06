from enum import Enum
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation

class ModelEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"

class ModelManager(object):

    @staticmethod
    def get_simulation_model(model_name : str, states):
        match model_name :
            case "Gol":
                return GOLSimulation(init_states=states)
            case "Lenia":
                return LeniaSimulation(init_states=states)
            
    @staticmethod
    def get_default_params(model_name : str):
        match model_name:
            case "Gol":
                return GOLSimulation.default_parameters
            case "Lenia":
                return LeniaSimulation.default_parameters

