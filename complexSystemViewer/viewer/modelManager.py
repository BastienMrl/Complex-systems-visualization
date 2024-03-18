from enum import Enum
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation
from simulation.models.flocking import FlockingSimulation

class ModelEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"
    FLOCKING = "Flocking"

class ModelManager(object):

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
                return GOLSimulation.default_rules
            case "Lenia":
                return LeniaSimulation.default_rules
            case "Flocking":
                return FlockingSimulation.default_rules
    
    @staticmethod
    def get_initialization_parameters(model_name : str):
        match model_name:
            case "Gol":
                return GOLSimulation.initialization_parameters
            case "Lenia":
                return LeniaSimulation.initialization_parameters
            case "Flocking":
                return FlockingSimulation.initialization_parameters

