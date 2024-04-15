from abc import ABC, abstractmethod, abstractproperty
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .param import *
from .interaction import *
import numpy as np
import time
import copy


class Simulation(ABC):   

    

    """Abstract super-class to extend if you want to add a new simulation

    :param list[State] init_states: Initial states of the simulation
    :param list[Param] rules: parameters required to run the simulation
    """

    
    @property
    @abstractmethod
    def initialization_parameters(self):
        """Abstract attribute containing list of Param that will be exposed to the user and set before the simulation starts
    
        :rtype: list[Param]
        """
        pass
    
    
    
    @property
    @abstractmethod
    def default_rules(self):
        """Abstract attribute : list of Param that will be exposed to the user and can be modified anytime during the simulation
    
        :rtype: list[Param]
        """
        pass
    
    
    def __init__(self, init_states : State = None, rules : list[Param]  = None, needJSON : bool = True): 
        self.NEED_JSON = needJSON
        self.HISTORY_SIZE = 5
        self._history_idx = 0
        self.past_states : list[State] = [None] * self.HISTORY_SIZE
        self.current_states : State = None
        self.rules : list[Param] = None
        self.width : int = None
        self.height : int = None
        self.as_json : list[list[float]] = None
        self.init_param : list[Param] = None
        if init_states != None :
             self.current_states = init_states
        if rules != None :
            self.rules = rules

        self.interactions : list[Interaction] = None
            
    @abstractmethod
    def initSimulation(self, init_states = None, rules = None, init_param = None):
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

    def getRules(self): 
        """Access the rules parameters
        
        :returns: the exposed parameters used to run the simulation
        :rtype: list[Param]
        """
        return self.rules
    
    def getRuleById(self, id:str):
        """Access the current value of a paramer used to run the simulation
        
        :param id: Name of the paramer 
        
        :returns: The value of the parameter 
        """
        for p in self.rules:
            if p.id_param == id:
                return p.value
        return None
    
    def updateRule(self, json):
        """Set the value of a paramer used to run the simulation
        
        :param json: a dictionary representation of the parameter"
        """
        if (json == None):
            return
        for p in self.rules:
            if p.id_param == json["paramId"]:
                p.set_param(json)
        
    def applyInteraction(self, id : str, mask : jnp.ndarray):
        """Apply the specified interaction to the current states of the simulation
        
        :param str id: Id of the interaction
        :param jnp.ndarray mask: A 2D mask of floats used to apply the interaction
        """
        interaction : None | Interaction = None
        for element in self.interactions:
            if element.id == id:
                interaction = element
        if interaction == None :
            return

        interaction.apply(mask, self.current_states)
        if (self.NEED_JSON):
            self.to_JSON_object()

    def set_current_state_from_id(self, id : int):
        for state in self.past_states:
            if (state.id == id):
                current_id = self.current_states.id
                self.current_states = state
                self.current_states.id = current_id + 1
                break

    def newStep(self):    
        if (self.NEED_JSON):
            self.past_states[self._history_idx] = copy.deepcopy(self.current_states)
            self._history_idx = (self._history_idx + 1) % self.HISTORY_SIZE
        self._step()
        self.current_states.id += 1
        if (self.NEED_JSON):
            self.to_JSON_object()

    def get_rules_param(self, id : str) -> Param | None:
        for p in self.rules:
            if p.id_param == id:
                return p
        return None        

    def get_init_param(self, id : str) -> Param | None:
        for p in self.init_param:
            if p.id_param == id:
                return p
        return None