from abc import ABC, abstractmethod, abstractproperty
import jax.numpy as jnp
from .param import *
from .state import *
import copy
from typing import Callable




class Simulation(ABC):   

    

    """Abstract super-class to extend if you want to add a new simulation

    :param list[State] init_states: Initial states of the simulation
    :param list[Param] rules: parameters required to run the simulation
    """

    
    # @property
    # @abstractmethod
    # def initialization_parameters(self):
    #     """Abstract attribute containing list of Param that will be exposed to the user and set before the simulation starts
    
    #     :rtype: list[Param]
    #     """
    #     pass
    
    
    
    # @property
    # @abstractmethod
    # def default_rules(self):
    #     """Abstract attribute : list of Param that will be exposed to the user and can be modified anytime during the simulation
    
    #     :rtype: list[Param]
    #     """
    #     pass
    
    
    def __init__(self, params : SimulationParameters, needJSON : bool = True): 
        self.NEED_JSON = needJSON
        self.HISTORY_SIZE = 5
        self._history_idx = 0
        self.past_states : list[State] = [None] * self.HISTORY_SIZE
        self.current_states : State = None
        self.rules : list[Param] = None
        self.width : int = None
        self.height : int = None
        self.as_json : list[list[float]] = None
        self.params : SimulationParameters = params
        self.interactions : SimulationInteractions = None
            
    @abstractmethod
    def init_simulation(self, params : SimulationParameters = None):
        """Method called before the simulation starts. It is expected to set the initial states of the simulation.

        :param list[State] init_states: Optional states for the simulation 
        :param list[Param] init_states: Optional parameters for the initializatin 
        """
        pass

    @abstractmethod
    def _step(self) : 
        """Method executing a state of the simulation. It is expected to update the attribute :py:attr:current_states.
        """
        pass

    
    def to_JSON_object(self) :
        """Converts the current states of the simulation to JSON-serializable python object. Used for convenience when manipulating states from outside of the simualtion module.
        
        :returns: a JSON-serializable representation of the current states of the simulation.
        """
        tsl = self.current_states.to_JSON_object()
        self.as_json = tsl

        
    def apply_interaction(self, id : str, mask : jnp.ndarray):
        """Apply the specified interaction to the current states of the simulation
        
        :param str id: Id of the interaction
        :param jnp.ndarray mask: A 2D mask of floats used to apply the interaction
        """
        self.interactions.apply_interaction(id, mask, self)
        if (self.NEED_JSON):
            self.to_JSON_object()

    def set_current_state_from_id(self, id : int):
        for state in self.past_states:
            if (state.id == id):
                current_id = self.current_states.id
                self.current_states = state
                self.current_states.id = current_id + 1
                break

    def new_step(self):    
        if (self.NEED_JSON):
            self.past_states[self._history_idx] = copy.deepcopy(self.current_states)
            self._history_idx = (self._history_idx + 1) % self.HISTORY_SIZE
        self._step()
        self.current_states.id += 1
        if (self.NEED_JSON):
            self.to_JSON_object()



class Interaction():
    def __init__(self, id : str, apply_fct : Callable[[jnp.ndarray, Simulation], None]):
        self.id = id
        self.apply_fct = apply_fct

    def apply(self, mask : jnp.ndarray, states : list[State]):
        self.apply_fct(mask, states)

class SimulationInteractions():
    def __init__(self):
        self.interactions : dict[str, Callable[[jnp.ndarray, Simulation]]]

    def apply_interaction(self, id : str, mask : jnp.ndarray, simulation : Simulation):
        
        print("id = ", id)

        if (not id in self.interactions.keys()):
            return
        
        print("execution")
        
        self.interactions[id](mask, simulation)

    def get_names(self):
        return self.interactions.keys()

    def _get_mask_for_only_one_channel(self, mask : jnp.ndarray, channel : int, nb_channels : int):
        mask = jnp.expand_dims(mask, 2)
        shape = list(mask.shape)

        minus_one = jnp.subtract(jnp.zeros(shape), jnp.ones(shape))
        new_mask = mask if channel == 0 else minus_one
        if (nb_channels > 1):
            for k in range(1, nb_channels):
                if (k == channel):
                    new_mask = jnp.dstack((new_mask, mask))
                else:
                    new_mask = jnp.dstack((new_mask, minus_one))
        return new_mask
    
    def _set_channel_value_with_mask(self, channel : int, mask : jnp.ndarray, simulation : Simulation):
        nb_channels = simulation.current_states.grid.shape[-1]
        if (nb_channels <= channel):
            return

        new_mask = self._get_mask_for_only_one_channel(mask, channel, nb_channels)

        simulation.current_states.grid = jnp.where(new_mask >= 0, new_mask, simulation.current_states.grid)