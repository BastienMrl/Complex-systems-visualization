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
                    
                    await self.initLenia(text_data_json["params"])
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
                    await self.applyInteraction(text_data_json["mask"])
                    

    async def emptyGrid(self, nbInstances):
        row = int(math.sqrt(nbInstances))
        grid = [0] * row * row
        xy = jnp.indices([row, row], dtype=jnp.float32)
        offset = -(row - 1) / 2.
        x = ((xy[1].reshape(row * row) + offset) ).tolist()
        y = ((xy[0].reshape(row * row) + offset) ).tolist()
        data = [x, y, grid]
        await self.send(bytes_data=orjson.dumps(data))


    async def initGOL(self, nbInstances):
        key = jax.random.PRNGKey(1701)
        nb = nbInstances
        row = int(math.sqrt(nb))
        grid = jax.random.uniform(key, (1, row, row, 1))
        grid = jnp.round(grid)
        grid = jnp.transpose(grid, [0, 3, 1, 2])

        #print("prep state")
        state = GridState(grid)

        #print("prep sim")
        gol = GOLSimulation(init_states=[state])
        
        self.sim = gol


    async def initLenia(self, nbInstances):
        seed = 10
        key = jax.random.PRNGKey(seed)
        params_seed, state_seed = jax.random.split(key)
        SX = SY = int(math.sqrt(nbInstances))
        mx, my = SX//2, SY//2 # center coordinated
        A0 = jnp.zeros((SX, SY, 1)).at[mx-20:mx+20, my-20:my+20, :].set(
            jax.random.uniform(state_seed, (40, 40, 1))
        )

        #print("prep state")
        state = GridState(A0)

        params = LeniaSimulation.default_parameters

        for p in params :
            if p.id_param == "gridSize" :
                p.value = SX

        #print("prep sim")
        lenia = LeniaSimulation(init_states=[state], init_params=params)
        self.sim = lenia


    async def sendOneStep(self):
        t0 = time.time()
        await self.send(bytes_data=orjson.dumps(self.sim.to_JSON_object()))
        #print("Data sent - ", 1000*(time.time()-t0), "ms\n")
        self.sim.step()



    async def updateRules(self, params):
        rules = orjson.loads(params)
        for rule in rules:
            match rule:
                case "birth":
                    parameter : RangeIntParam = self.sim.getParamById("birth")
                    parameter.min_param.value = rules[rule][0]
                    parameter.max_param.value = rules[rule][1]
                case "survival":
                    parameter : RangeIntParam = self.sim.getParamById("survival")
                    parameter.min_param.value = rules[rule][0]
                    parameter.max_param.value = rules[rule][1]


    async def applyInteraction(self, mask):
        mask_jnp = jnp.array(mask).reshape(self.sim.width, self.sim.height)
        self.sim.applyInteraction("toLife", mask_jnp)
        

    
