from enum import Enum
import copy as cp
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import LeniaSimulation
from simulation.models.flocking import FlockingSimulation
from simulation.models.diffusion import DiffusionSimulation
from simulation.models.physarum_agent import PhysarumAgentSimulation
from simulation.models.ant_colony import AntColony

class SimulationEnum(Enum):
    GOL = "Gol"
    LENIA = "Lenia"
    FLOCKING = "Flocking"
    DIFFUSION = "Diffusion"
    PHYSARUM_AGENTS = "Physarum agents"
    ANT_COLONY = "Ant colony"

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
                return AntColony()
            
            
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
                return cp.deepcopy(AntColony.default_rules)
    
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
                return cp.deepcopy(AntColony.initialization_parameters)

