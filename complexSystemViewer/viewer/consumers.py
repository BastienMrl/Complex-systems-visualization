import orjson
import jax.numpy as jnp
import jax.image as jimage
from .simulation_manager import SimulationManager
from channels.generic.websocket import AsyncWebsocketConsumer
from simulation.simulation import Simulation
import math
import copy as cp

class ViewerConsumerV2(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isConnected = False
        self.sim : Simulation = None
        self.params = None   
    
    async def connect(self):
        await self.accept()
        self.params = SimulationManager.get_parameters("Gol")
        self.next_params_set = SimulationManager.get_parameters("Gol")
        self.sim = SimulationManager.get_simulation_model("Gol", self.params)
        self.isConnected = True
    
    async def disconnect(self, close_code):
        self.isConnected = False

    async def receive(self, text_data=None):
        text_data_json = orjson.loads(text_data)
        message = text_data_json["message"]
        match message:
            case "RequestData":
                if self.isConnected :
                    await self.sendOneStep()
            case "ResetSimulation":
                if self.isConnected:
                    await self.resetSimulation()
            case "ChangeSimulation":
                if self.isConnected:
                    self.sim = None
                    await self.initNewSimulation(text_data_json["simuName"])       
            case "UpdateInitParams":
                    await self.updateInitParams(text_data_json["params"])
            case "UpdateRule":
                if self.isConnected:
                    await self.updateRule(text_data_json["params"])
            case "ApplyInteraction":
                if self.isConnected:
                    await self.applyInteraction(text_data_json["mask"], text_data_json["id"], text_data_json["interaction"])

    async def sendOneStep(self):
        await self.send(bytes_data=orjson.dumps(self.sim.as_json))
        self.sim.newStep()

    async def updateInitParams(self, params):
        json = orjson.loads(params)
        self.next_params_set.update_init_param(json)

    async def updateRule(self, params):
        self.sim.params.update_rules_param(orjson.loads(params))

    async def initNewSimulation(self, name):
        self.params = SimulationManager.get_parameters(name)
        self.next_params_set = SimulationManager.get_parameters(name)
        self.sim = SimulationManager.get_simulation_model(name, self.params)
        await self.sendOneStep()    

    async def resetSimulation(self):
        self.params.set_init_from_list(self.next_params_set.get_init_parameters())
        self.next_params_set = cp.deepcopy(self.next_params_set)
        self.sim.initSimulation(self.params)
        await self.sendOneStep()

    async def applyInteraction(self, mask : list[float], stateId : int, interaction : str):
        mask_width = int(math.sqrt(len(mask)))
        
        self.sim.set_current_state_from_id(stateId)
        mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
        mask_jnp = mask_jnp.reshape((mask_width, mask_width))
        mask_jnp = jimage.resize(mask_jnp, (self.sim.width, self.sim.height), "linear")
        self.sim.applyInteraction(interaction, mask_jnp)
        await self.sendOneStep()        

    
