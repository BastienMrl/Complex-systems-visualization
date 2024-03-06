from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation

class ModelManager(object):

    @staticmethod
    def get_simulation_model(model_name : str, states):
        match model_name :
            case "gol":
                return GOLSimulation(init_states=states)
            case "lenia":
                return LeniaSimulation(init_states=states)
            
    @staticmethod
    def get_default_params(model_name : str):
        match model_name:
            case "gol":
                return GOLSimulation.default_parameters
            case "lenia":
                return LeniaSimulation.default_parameters

