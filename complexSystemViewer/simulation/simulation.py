from abc import ABC, abstractmethod, abstractproperty
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .param import *
from .interaction import *
import numpy as np
import time


class Simulation(ABC):   

    def __init__(self, init_states = None, rules  = None): 
        self.current_states = None
        self.rules = None
        self.width = None
        self.height = None
        # if init_states != None :
        #     self.current_states = init_states
        # else : 
        #     raise ValueError("Initial State can't be None")
        
        # self.width = init_states[0].width
        # self.height = init_states[0].height
        # for state in init_states : 
        #     if not jnp.isclose(state.width, self.width) or not jnp.isclose(state.height, self.height):
        #         raise ValueError("States of a simulation must be of same size")
        

        if rules != None :
            self.rules = rules
        else :
            pass
            #raise ValueError("Initial parameters can't be None")
        self.interactions : list[Interaction] = None
            
    @abstractmethod
    def initSimulation(self, init_states = None, rules = None, init_param = None):
        pass

    @abstractmethod
    def step(self) : 
        pass

    @abstractmethod
    def set_current_state_from_array(self, new_state):
        pass

    def to_JSON_object(self) :
        t0 = time.time()
        """ particules = self.current_states[0].particles
        
        row_gen = (np.append(np.array([p.pos_x, p.pos_y], dtype=float),p.values) for p in particules)

        arr = np.fromiter(row_gen, object)
        
        tsl = np.transpose(np.stack(arr)).tolist() """
        tsl = self.current_states[0].to_JSON_object()
        #print("json obj ok - ", 1000*(time.time()-t0), "ms\n")
        return tsl

    def getRules(self): 
        return self.rules
    
    def getRuleById(self, id:str):
        for p in self.rules:
            if p.id_param == id:
                return p.value
        return None
    
    def updateRule(self, json):
        for p in self.rules:
            if p.id_param == json["paramId"]:
                p.set_param(json)
        
    def applyInteraction(self, id : str, mask : jnp.ndarray):
        interaction : None | Interaction = None
        for element in self.interactions:
            if element.id == id:
                interaction = element
        if interaction == None :
            return

        interaction.apply(mask, self.current_states)

        
        
        
        
    