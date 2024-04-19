from abc import ABC, abstractmethod 
from enum import Enum



from .param import *

# from viewer.simulation_manager import 




class Paramtype(Enum):
    NUMBERRANGE = 'NR'
    NUMBERVALUE = 'NV'
    COLORVALUE = 'CV'
    SELECTIONVALUE = 'SV'
    BOOLVALUE='BV'



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
    def set_param(self, json): 
        pass

    def set_id(self, id : str):
        self.id_param = id

    def set_name(self, name : str):
        self.name = name

        

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
    
class BoolParam(Param):
    def __init__(self, id_p: str, name: str, default_value : bool):
        super().__init__(id_p, Paramtype.BOOLVALUE, name)
        self.value = default_value
        self.default_value = default_value
    
    def get_param(self):
        superParam = super().get_param()
        superParam.update({
            "defaultValue" : self.default_value,
        })
        return superParam

    def set_param(self, json):
        self.value = json["value"]

class SelectionParam(Param):
    def __init__(self, id_p : str, name : str, options : list[str] = None, default_idx : int = 0):
        super().__init__(id_p, Paramtype.SELECTIONVALUE, name)
        self.options = options
        self.default_idx = default_idx
        self.value = 0

    def get_param(self):
        param = super().get_param()
        param.update({
            "defaultValue" : self.default_idx,
            "options" : self.options,
        })
        return param

    def set_param(self, json):
        self.value = json["value"]





class SimulationParameters(ABC):

    def __init__(self, id_prefix : str = "default"):
        self.id_prefix : str = id_prefix
        self._rules_param : list[Param] = []
        self._init_param : list[Param] = []
        self.type : type = SimulationParameters

    def set_prefix(self, prefix : str):
        old_id = self._get_id_prefix()
        old_name = self._get_name_prefix()
        self.id_prefix = prefix
        for p in self._rules_param:
            new_id = self._get_id_prefix() + self._get_id_without_prefix(p.id_param, old_id)
            new_name = self._get_name_prefix() + self._get_name_without_prefix(p.id_param, old_name)
            
            p.set_id(new_id)
            p.set_name(new_name)
            
        for p in self._init_param:
            new_id = self._get_id_prefix() + self._get_id_without_prefix(p.id_param, old_id)
            new_name = self._get_name_prefix() + self._get_name_without_prefix(p.id_param, old_name)

            p.set_id(new_id)
            p.set_name(new_name)

    def update_rules_param(self, json : dict[str, str | float | int]):
        if (json == None):
            return
        if (not json["paramId"].startswith(self.id_prefix)):
            return
        for i, p in enumerate(self.get_rules_paramaters()):
            if p.id_param == json["paramId"]:
                p.set_param(json)
                self.rule_param_value_changed(i, p)

    def update_init_param(self, json : dict[str, str | float | int]):
        if (json == None):
            return
        if (not json["paramId"].startswith(self.id_prefix)):
            return
        for i, p in enumerate(self.get_init_parameters()):
            if p.id_param == json["paramId"]:
                p.set_param(json)
                self.init_param_value_changed(i, p)
        
    @abstractmethod
    def rule_param_value_changed(self, idx : int, param : Param) -> None:
        pass

    @abstractmethod
    def init_param_value_changed(self, idx : int, param : Param) -> None:
        pass

    def set_all_params(self) -> None:
        for i, p in enumerate(self.get_init_parameters()):
            self.init_param_value_changed(i, p)
        for i, p in enumerate(self.get_rules_paramaters()):
            self.rule_param_value_changed(i, p)

    def set_init_from_list(self, source : list[Param]):
        for s in source:
            for i in range(len(self._init_param)):
                if self._init_param[i].id_param == s.id_param:
                    self._init_param[i] = s
        for i, p in enumerate(self.get_init_parameters()):
            self.init_param_value_changed(i, p)


    def _get_name_prefix(self) -> str:
        if (self.id_prefix != "default"):
            return f"({self.id_prefix}) "
        else:
            return "" 
        
    def _get_id_prefix(self) -> str:
        return self.id_prefix + "-"
    
    def _get_id_without_prefix(self, id : str, prefix : str) -> str:
        return id.removeprefix(prefix)

    def _get_name_without_prefix(self, name : str, prefix : str) -> str:
        return name.removeprefix(prefix)

    def get_rules_paramaters(self) -> list[Param]:
        return self._rules_param
    
    def get_init_parameters(self) -> list[Param]:
        return self._init_param
