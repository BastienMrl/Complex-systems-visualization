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
                return cp.deepcopy(GOLSimulation.get_rules())
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.get_rules())
            case "Flocking":
                return cp.deepcopy(FlockingSimulation.get_rules())
            case "Diffusion":
                return cp.deepcopy(DiffusionSimulation.get_rules())
            case "Physarum agents":
                return cp.deepcopy(PhysarumAgentSimulation.get_rules())
            case "Ant colony":
                return cp.deepcopy(AntColony.get_rules())
    
    @staticmethod
    def get_initialization_parameters(model_name : str):
        match model_name:
            case "Gol":
                return cp.deepcopy(GOLSimulation.get_initialization())
            case "Lenia":
                return cp.deepcopy(LeniaSimulation.get_initialization())
            case "Flocking":
                return cp.deepcopy(FlockingSimulation.get_initialization())
            case "Diffusion":
                return cp.deepcopy(DiffusionSimulation.get_initialization())
            case "Physarum agents":
                return cp.deepcopy(PhysarumAgentSimulation.get_initialization())
            case "Ant colony":
                return cp.deepcopy(AntColony.get_initialization())

