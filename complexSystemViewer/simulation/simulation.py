from abc import ABC, abstractmethod, abstractproperty
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .param import *
from .interaction import *
import numpy as np
import time


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
    
    #: A list of State to be updated at each step. This attibute can contain a single state as long it is in a single element list
    current_states = None

    def __init__(self, init_states : list[State] = None, rules : list[Param]  = None): 
        self.current_states = None
        self.rules = None
        self.width = None
        self.height = None
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
    def step(self) : 
        """Method executing a state of the simulation. It is expected to update the attribute :py:attr:current_states.
        """
        pass

    @abstractmethod
    def set_current_state_from_array(self, new_state):
        """Set the states of the simulation form an arbitrary array-like representation. Used for convenience when handlind external data. 

        :param new_state: Reprensnetation of states of the simulation
        :type new_state: list | ndarray
        """
        pass

    
    def to_JSON_object(self) :
        """Converts the current states of the simulation to JSON-serializable python object. Used for convenience when manipulating states from outside of the simualtion module.
        
        :returns: a JSON-serializable representation of the current states of the simulation.
        """
        t0 = time.time()
        tsl = self.current_states[0].to_JSON_object()
        #print("json obj ok - ", 1000*(time.time()-t0), "ms\n")
        return tsl

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

        
        
        
        
    