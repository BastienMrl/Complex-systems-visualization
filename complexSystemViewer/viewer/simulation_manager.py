from enum import Enum
import copy as cp
from simulation.models.game_of_life import *
from simulation.models.lenia import *
# from simulation.models.flocking import *
from simulation.models.diffusion import *
from simulation.models.physarum_agent import *
from simulation.models.ant_colony import *
from simulation.models.gray_scott import *

class SimulationEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"
    FLOCKING = "Flocking"
    DIFFUSION = "Diffusion"
    PHYSARUM_AGENTS = "Physarum agents"
    ANT_COLONY = "Ant colony"
    MULTIPLE_PHYSARUM = "Multiple physarum"
    PHYSARUM_LENIA = "Physarum Lenia"
    GRAY_SCOTT_PHYSARUM = "Physarum Gray-Scott"
    GRAY_SCOTT = "Gray-Scott Diffusion-Reaction"

    @classmethod
    def get_list(cls):
        return [e.value for e in cls]
    
    @classmethod
    def get_class(cls, value):
        match value:
            case cls.GOL : return GOLSimulation
            case cls.LENIA : return LeniaSimulation
            # case cls.FLOCKING : return FlockingSimulation
            case cls.DIFFUSION : return DiffusionSimulation
            case cls.PHYSARUM_AGENTS : return PhysarumAgentSimulation
            case cls.ANT_COLONY : return AntColonySimulation
            case cls.GRAY_SCOTT_PHYSARUM : return PhysarumGrayScott
            case cls.GRAY_SCOTT : return GrayScottSimulation


class SimulationManager(object):

    @staticmethod
    def get_simulation_model(model_name : str, params : SimulationParameters):
        match model_name :
            case "Gol":
                return GOLSimulation(params)
            case "Lenia":
                return LeniaSimulation(params)
            # case "Flocking":
            #     return FlockingSimulation(params)
            case "Diffusion":
                return DiffusionSimulation(params)
            case "Physarum agents":
                return PhysarumAgentSimulation(params)
            case "Ant colony":
                return AntColonySimulation(params)
            case "Physarum Lenia":
                return PhysarumLeniaSimulation(params)
            case "Multiple physarum":
                return MultiplePhysarumSimulation(params)
            case "Physarum Gray-Scott":
                return PhysarumGrayScott(params)
            case "Gray-Scott Diffusion-Reaction":
                return GrayScottSimulation(params)
            
            
    @staticmethod
    def get_default_rules(model_name : str):
        match model_name:
            case "Gol":
                return GOLParameters().get_rules_parameters()
            case "Lenia":
                return LeniaParameters().get_rules_parameters()
            # case "Flocking":
            #     return FlockingParameters().get_rules_parameters()
            case "Diffusion":
                return DiffusionParameters().get_rules_parameters()
            case "Physarum agents":
                return PhysarumAgentParameters().get_rules_parameters()
            case "Ant colony":
                return AntColonySimulation.params.get_rules_parameters()
            case "Physarum Lenia":
                return PhysarumLeniaSimulation.params.get_rules_parameters()
            case "Multiple physarum":
                return MultiplePhysarumSimulation.params.get_rules_parameters()
            case "Physarum Gray-Scott":
                return PhysarumGrayScott.params.get_rules_parameters()
            case "Gray-Scott Diffusion-Reaction":
                return GrayScottParameters().get_rules_parameters()
    
    @staticmethod
    def get_initialization_parameters(model_name : str):
        match model_name:
            case "Gol":
                return GOLParameters().get_init_parameters()
            case "Lenia":
                return LeniaParameters().get_init_parameters()
            # case "Flocking":
            #     return FlockingParameters().get_init_parameters()
            case "Diffusion":
                return DiffusionParameters().get_init_parameters()
            case "Physarum agents":
                return PhysarumAgentParameters().get_init_parameters()
            case "Ant colony":
                return AntColonySimulation.params.get_init_parameters()
            case "Physarum Lenia":
                return PhysarumLeniaSimulation.params.get_init_parameters()
            case "Multiple physarum":
                return MultiplePhysarumSimulation.params.get_init_parameters()
            case "Physarum Gray-Scott":
                return PhysarumGrayScott.params.get_init_parameters()
            case "Gray-Scott Diffusion-Reaction":
                return GrayScottParameters().get_init_parameters()

    @staticmethod
    def get_parameters(model_name : str):
        match model_name:
            case "Gol":
                return GOLParameters()
            case "Lenia":
                return LeniaParameters()
            # case "Flocking":
            #     return FlockingParameters()
            case "Diffusion":
                return DiffusionParameters()
            case "Physarum agents":
                return PhysarumAgentParameters()
            case "Ant colony":
                return AntColonySimulation.params
            case "Physarum Lenia":
                return PhysarumLeniaSimulation.params
            case "Multiple physarum":
                return MultiplePhysarumSimulation.params
            case "Physarum Gray-Scott":
                return PhysarumGrayScott.params
            case "Gray-Scott Diffusion-Reaction":
                return GrayScottParameters()
            
    @staticmethod
    def get_interactions_names(model_name : str):
         match model_name:
            case "Gol":
                return GOLInteractions().get_names()
            case "Lenia":
                return LeniaInteractions().get_names()
            # case "Flocking":
            #     return SimulationInteractions().get_names()
            case "Diffusion":
                return DiffusionInteractions().get_names()
            case "Physarum agents":
                return SimulationInteractions().get_names()
            case "Ant colony":
                return AntColonyInteractions().get_names()
            case "Physarum Lenia":
                return AntColonyInteractions().get_names()
            case "Multiple physarum":
                return AntColonyInteractions().get_names()
            case "Physarum Gray-Scott":
                return AntColonyInteractions().get_names()
            case "Gray-Scott Diffusion-Reaction":
                return GrayScottInteractions().get_names()
