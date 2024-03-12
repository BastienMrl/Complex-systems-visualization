import orjson

import time
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import math
from channels.generic.websocket import AsyncWebsocketConsumer
from simulation.state import State, GridState
from simulation.models.game_of_life import GOLSimulation
from simulation.param import *
from simulation.models.lenia import  LeniaSimulation

import time


class ViewerConsumerV2(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isConnected = False
        
        #####
        self.sim = None
    
    async def connect(self):
        await self.accept()
        self.isConnected = True
    
    async def disconnect(self, close_code):
        self.isConnected = False

    async def receive(self, text_data=None):
        text_data_json = orjson.loads(text_data)
        message = text_data_json["message"]
        match message:
            case "Start":
                if self.isConnected:
                    await self.initLenia()
            case "Stop":
                if self.isConnected:
                    self.sim = None
            case "RequestData":
                if self.isConnected :
                    await self.sendOneStep()
            case "EmptyGrid":
                if self.isConnected:
                    await self.emptyGrid(text_data_json["params"])
            case "ChangeRules":
                if self.isConnected:
                    await self.updateRules(text_data_json["params"])
            case "ApplyInteraction":
                if self.isConnected:
                    t = time.time()
                    await self.applyInteraction(text_data_json["mask"], text_data_json["currentStates"])
                    print("time = ", (1000 * (time.time() - t)), "ms")
                    

    async def emptyGrid(self, nbInstances):
        row = int(math.sqrt(nbInstances))
        grid = [0] * row * row
        xy = jnp.indices([row, row], dtype=jnp.float32)
        offset = -(row - 1) / 2.
        x = ((xy[1].reshape(row * row) + offset) ).tolist()
        y = ((xy[0].reshape(row * row) + offset) ).tolist()
        data = [x, y, grid]
        await self.send(bytes_data=orjson.dumps(data))


    async def initGOL(self):
        self.sim = GOLSimulation()


    async def initLenia(self):
        self.sim = LeniaSimulation()


    async def sendOneStep(self):
        t0 = time.time()
        await self.send(bytes_data=orjson.dumps(self.sim.to_JSON_object()))
        #print("Data sent - ", 1000*(time.time()-t0), "ms\n")
        self.sim.step()



    async def updateRules(self, params):
        json = orjson.loads(params)
        self.sim.updateParam(json)
        #for rule in rules:
        #    match rule:
        #        case "birth":
        #            parameter : RangeIntParam = self.sim.getParamById("birth")
        #            parameter.min_param.value = rules[rule][0]
        #            parameter.max_param.value = rules[rule][1]
        #        case "survival":
        #            parameter : RangeIntParam = self.sim.getParamById("survival")
        #            parameter.min_param.value = rules[rule][0]
        #            parameter.max_param.value = rules[rule][1]


    async def applyInteraction(self, mask, currentValues):
        self.sim.set_current_state_from_array(currentValues)
        mask_jnp = jnp.array(mask).reshape(self.sim.width, self.sim.height)
        self.sim.applyInteraction("toLife", mask_jnp)
        

    
