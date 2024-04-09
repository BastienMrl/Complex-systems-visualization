import orjson
import jax.numpy as jnp
import jax.image as jimage
from .simulationManager import SimulationManager
from channels.generic.websocket import AsyncWebsocketConsumer
from simulation.simulation import Simulation
import math


class ViewerConsumerV2(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isConnected = False
        self.sim : Simulation = None
        self.init_parameters = None   
    
    async def connect(self):
        await self.accept()
        self.init_parameters = SimulationManager.get_initialization_parameters("Gol")
        self.sim = SimulationManager.get_simulation_model("Gol")
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
        for p in self.init_parameters:
            if p.id_param == json["paramId"]:
                p.set_param(json)

    async def updateRule(self, params):
        self.sim.updateRule(orjson.loads(params))

    async def initNewSimulation(self, name):
        self.init_parameters = SimulationManager.get_initialization_parameters(name)
        self.sim = SimulationManager.get_simulation_model(name)
        await self.sendOneStep()    

    async def resetSimulation(self):
        self.sim.initSimulation(init_param=self.init_parameters)
        await self.sendOneStep()

    async def applyInteraction(self, mask : list[float], stateId : int, interaction : str):
        mask_width = int(math.sqrt(len(mask)))
        
        self.sim.set_current_state_from_id(stateId)
        mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
        mask_jnp = mask_jnp.reshape((mask_width, mask_width))
        mask_jnp = jimage.resize(mask_jnp, (self.sim.width, self.sim.height), "linear")
        self.sim.applyInteraction(interaction, mask_jnp)
        await self.sendOneStep()        

    
