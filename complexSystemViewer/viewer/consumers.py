import orjson

import time
import jax.numpy as jnp
import math
from .modelManager import ModelManager
from channels.generic.websocket import AsyncWebsocketConsumer
from simulation.models.game_of_life import GOLSimulation
from simulation.models.lenia import  LeniaSimulation
from simulation.simulation import Simulation

import time


class ViewerConsumerV2(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isConnected = False
        self.sim : Simulation = None
    
    async def connect(self):
        self.isConnected = True
        self.sim = GOLSimulation()
        await self.accept()
    
    async def disconnect(self, close_code):
        self.isConnected = False

    async def receive(self, text_data=None):
        text_data_json = orjson.loads(text_data)
        message = text_data_json["message"]
        print(message)
        match message:
            case "Start":
                if self.isConnected:
                    await self.resetSimulation()
            case "Stop":
                if self.isConnected:
                    self.sim = None
            case "RequestData":
                if self.isConnected :
                    await self.sendOneStep()
            case "EmptyGrid":
                if self.isConnected:
                    await self.emptyGrid(text_data_json["params"])
            case "UpdateRules":
                if self.isConnected:
                    await self.updateRules(text_data_json["params"])
            case "ApplyInteraction":
                if self.isConnected:
                    t = time.time()
                    await self.applyInteraction(text_data_json["mask"], text_data_json["currentStates"])
            case "ChangeSimulation":
                if self.isConnected:
                    self.sim = None
                    await self.initNewSimulation(text_data_json["simuName"])
                    

    async def emptyGrid(self, nbInstances):
        row = int(math.sqrt(nbInstances))
        grid = [0] * row * row
        xy = jnp.indices([row, row], dtype=jnp.float32)
        offset = -(row - 1) / 2.
        x = ((xy[1].reshape(row * row) + offset) ).tolist()
        y = ((xy[0].reshape(row * row) + offset) ).tolist()
        data = [x, y, grid]
        await self.send(bytes_data=orjson.dumps(data))


    async def sendOneStep(self):
        t0 = time.time()
        await self.send(bytes_data=orjson.dumps(self.sim.to_JSON_object()))
        # print("Data sent - ", 1000*(time.time()-t0), "ms\n")
        self.sim.step()


    async def updateRules(self, params):
        self.sim.updateParam(orjson.loads(params))


    async def initNewSimulation(self, name):
        print(name)
        self.sim = ModelManager.get_simulation_model(name)
    
    async def resetSimulation(self):
        self.sim.initSimulation()


    async def applyInteraction(self, mask, currentValues):
        self.sim.set_current_state_from_array(currentValues)
        mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
        mask_jnp = mask_jnp.reshape(self.sim.width, self.sim.height)
        self.sim.applyInteraction("toLife", mask_jnp)
        await self.send(bytes_data=orjson.dumps(self.sim.to_JSON_object()))
        

    
