from abc import ABC, abstractmethod 
from enum import Enum
import jax.numpy as jnp
import jax.lax as lax
import jax.random
from .param import *
import numpy as np

class Paramtype(Enum):
    NUMBERRANGE = 'NR'
    NUMBERVALUE = 'NV'
    COLORVALUE = 'CV'
    SELECTIONVALUE = 'SV'



class Param(ABC): 
    #id_param = None
    #type_param : Paramtype = None
    #name  = None
    #default_value = None
    #min = None
    #max = None
    #step = None

    def __init__(self, id_p:str, type_p:Paramtype, name: str):
        self.id_param = id_p
        self.type_param = type_p
        self.name = name

    # @abstractmethod
    # def set_value(self, value):
    #     pass

    # @abstractmethod
    # def convert(self, value):
    #     pass

    # def get_value(self, value):
    #     return self.value
    
    def get_param(self):
        return {
            "paramId": self.id_param,
            "name": self.name,
            "type": str(self.type_param.value)
        }
        
    @abstractmethod
    def set_param(self, json): pass


        

class FloatParam(Param):

    def __init__(self, id_p , name, default_value : float, min_value:float = None, max_value:float = None, step:float = None):
        super().__init__(id_p, Paramtype.NUMBERVALUE, name)
        self.value = default_value
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def get_param(self):
        superParam = super().get_param()
        superParam.update({
            "defaultValue": self.default_value,
            "minValue": self.min_value,
            "maxValue": self.max_value,
            "step": self.step
        })
        return superParam
    
    def set_param(self, json):
        self.value = json["value"]
        
    
class IntParam(Param):
    def __init__(self, id_p , name, default_value : int, min_value:int = None, max_value:int = None, step:int = None):
        super().__init__(id_p, Paramtype.NUMBERVALUE, name)
        self.value = default_value
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step

    def get_param(self):
        superParam = super().get_param()
        superParam.update({
            "defaultValue": self.default_value,
            "minValue": self.min_value,
            "maxValue": self.max_value,
            "step": self.step
        })
        return superParam
    
    def set_param(self, json):
        self.value = json["value"]

class RangeFloatParam(Param):
    def __init__(self, id_p : str, name : str, min_param : FloatParam, max_param : FloatParam):
        super().__init__(id_p, Paramtype.NUMBERRANGE, name)
        self.min_param = min_param
        self.max_param = max_param

    def get_param(self):
        superParam = super().get_param()
        superParam.update({
            "minDefaultValue" : self.min_param.default_value,
            "maxDefaultValue" : self.max_param.default_value,
            "minMinimumValue" : self.min_param.min_value,
            "minMaximumValue" : self.min_param.max_value,
            "maxMinimumValue" : self.max_param.min_value,
            "maxMaximumValue" : self.max_param.max_value,
            "minStep" : self.min_param.step,
            "maxStep" : self.max_param.step,
        })
        return superParam
    
    def set_param(self, json):
        if json["subparam"] == "min":
            self.min_param.value = json["value"]
        elif json["subparam"] == "max":
            self.max_param.value = json["value"]
    
class RangeIntParam(Param):
    def __init__(self, id_p : str, name : str, min_param : IntParam, max_param : IntParam):
        super().__init__(id_p, Paramtype.NUMBERRANGE, name)
        self.min_param = min_param
        self.max_param = max_param

    def get_param(self):
        superParam = super().get_param()
        superParam.update({
            "minDefaultValue" : self.min_param.default_value,
            "maxDefaultValue" : self.max_param.default_value,
            "minMinimumValue" : self.min_param.min_value,
            "minMaximumValue" : self.min_param.max_value,
            "maxMinimumValue" : self.max_param.min_value,
            "maxMaximumValue" : self.max_param.max_value,
            "minStep" : self.min_param.step,
            "maxStep" : self.max_param.step,
        })
        return superParam
    
    def set_param(self, json):
        if json["subparam"] == "min":
            self.min_param.value = json["value"]
        elif json["subparam"] == "max":
            self.max_param.value = json["value"]