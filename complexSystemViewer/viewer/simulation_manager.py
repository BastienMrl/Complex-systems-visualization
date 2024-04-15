from enum import Enum
import copy as cp
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation
from simulation.models.flocking import FlockingSimulation
from simulation.models.diffusion import DiffusionSimulation
from simulation.models.physarum_agent import PhysarumAgentSimulation
from simulation.models.ant_colony import AntColonySimulation, PhysarumLeniaSimulation

class SimulationEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"
    FLOCKING = "Flocking"
    DIFFUSION = "Diffusion"
    PHYSARUM_AGENTS = "Physarum agents"
    ANT_COLONY = "Ant colony"
    PHYSARUM_LENIA = "Physarum Lenia"

    @classmethod
    def get_list(cls):
        return [e.value for e in cls]
    
    @classmethod
    def get_class(cls, value):
        match value:
            case cls.GOL : return GOLSimulation
            case cls.LENIA : return LeniaSimulation
            case cls.FLOCKING : return FlockingSimulation
            case cls.DIFFUSION : return DiffusionSimulation
            case cls.PHYSARUM_AGENTS : return PhysarumAgentSimulation
            case cls.ANT_COLONY : return AntColonySimulation


class SimulationManager(object):

    @staticmethod
    def get_simulation_model(model_name : str):
        match model_name :
            case "Gol":
                return GOLSimulation()
            case "Lenia":
                return LeniaSimulation()
            case "Flocking":
                return FlockingSimulation()
            case "Diffusion":
                return DiffusionSimulation()
            case "Physarum agents":
                return PhysarumAgentSimulation()
            case "Ant colony":
                return AntColonySimulation()
            case "Physarum Lenia":
                return PhysarumLeniaSimulation()
            
            
    @staticmethod
    def get_default_rules(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.default_rules)
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.default_rules)
            case "Flocking":
                return cp.deepcopy(FlockingSimulation.default_rules)
            case "Diffusion":
                return cp.deepcopy(DiffusionSimulation.default_rules)
            case "Physarum agents":
                return cp.deepcopy(PhysarumAgentSimulation.default_rules)
            case "Ant colony":
                return cp.deepcopy(AntColonySimulation.default_rules)
            case "Physarum Lenia":
                return cp.deepcopy(PhysarumLeniaSimulation.default_rules)
    
    @staticmethod
    def get_initialization_parameters(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.initialization_parameters)
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.initialization_parameters)
            case "Flocking":
                return cp.deepcopy(FlockingSimulation.initialization_parameters)
            case "Diffusion":
                return cp.deepcopy(DiffusionSimulation.initialization_parameters)
            case "Physarum agents":
                return cp.deepcopy(PhysarumAgentSimulation.initialization_parameters)
            case "Ant colony":
                return cp.deepcopy(AntColonySimulation.initialization_parameters)
            case "Physarum Lenia":
                return cp.deepcopy(PhysarumLeniaSimulation.initialization_parameters)

