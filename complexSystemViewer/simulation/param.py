from abc import ABC, abstractmethod 
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .parameter import *
import numpy as np
class Param(ABC): 
    id_param : String = None
    type_param : type = None
    name : String = None
    value = None
     

    def __init__(self, id_p : string, type_p:type, name:string, value):
        self.id_param = id_p
        self.type_param = type_p
        self.name = name
        self.value = value

    @abstractmethod
    def set_value(self, value):
        pass

    @abstractmethod
    def convert(self, value):
        pass

    def get_value(self, value):
        return self.value

    def to_JSON_object(self):
        obj : {
            'id' : self.id_param,
            'type' : str(self.type_param).split("\'")[1],
            'name' : self.name,
            'value' : self.value
        }


        

class IntParam(Param):     

    def __init__(self, id_p : string, name:string, value = 0):
        super().__init__(id_p, int, name, value)

    def convert(self, value):
        return int(value) #unhandled on purpose

    def set_value(self, value):
        value = convert(value)

    
    def get_value(self, value):
        return self.value