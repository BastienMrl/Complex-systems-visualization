import json
import asyncio
import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import math
from channels.generic.websocket import AsyncWebsocketConsumer
from simulation.state import State, GridState
from simulation.models.game_of_life import GOLSimulation


class ViewerConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_task = None
    
    async def connect(self):
        await self.accept()
        self.simulation_task = None
    
    async def disconnect(self, close_code):
        if self.simulation_task:
            self.simulation_task.cancel()

    async def receive(self, text_data=None):
        print("receive")
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        if message == "Start":
            print("Strat")
            if self.simulation_task is None or self.simulation_task.done():
                key = jax.random.PRNGKey(1701)
                nb = text_data_json["params"]
                row = int(math.sqrt(nb))
                grid = jax.random.uniform(key, (1, row, row, 1))
                grid = jnp.round(grid)
                grid = jnp.transpose(grid, [0, 3, 1, 2])

                print("prep state")
                state = GridState(grid)

                print("prep sim")
                gol = GOLSimulation(init_states=[state])
                
                print("prep sim done")
                self.simulation_task = asyncio.create_task(self.sendGameOfLife(gol))
        elif message == "Stop":
            if self.simulation_task:
                self.simulation_task.cancel()
                self.simulation_task = None

    async def sendRandomBool(self, nb):
        def generate_states():
            l = []
            for i in range(nb):
                l.append(round(random.random()))
            return l
        
        try:
            while True:
                await self.send(text_data=json.dumps(generate_states()))
                await asyncio.sleep(0.01)  # Contrôle la vitesse de la simulation
        except asyncio.CancelledError:
            pass  # La tâche a été annulée, arrêter la simulation proprement


    async def sendGameOfLife(self,sim):   
        print("send")
        try:
            while True:
                print("step")
                await self.send(text_data=json.dumps(sim.to_JSON_object()))
                sim.step()
                await asyncio.sleep(0.15)  # Contrôle la vitesse de la simulation
        except asyncio.CancelledError:
            print("cancelled")
            pass  # La tâche a été annulée, arrêter la simulation proprement


class ViewerConsumerV2(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isConnected = False
        self.isRunning = False
        self.kernel = None
        self.grid = None
        self.x = None
        self.y = None
    
    async def connect(self):
        await self.accept()
        self.isConnected = True
    
    async def disconnect(self, close_code):
        self.isConnected = False

    async def receive(self, text_data=None):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        match message:
            case "Start":
                if self.isConnected:
                    self.isRunning = True
                    await self.initGOL(text_data_json["params"])
            case "Stop":
                if self.isConnected:
                    self.isRunning = False
                    self.grid = None
                    self.kernel = None
                    self.x = None
                    self.y = None
            case "RequestData":
                if self.isConnected and self.isRunning:
                    await self.sendOneStepGOL()
            case "EmptyGrid":
                if self.isConnected:
                    await self.emptyGrid(text_data_json["params"])

    async def emptyGrid(self, nbInstances):
        row = int(math.sqrt(nbInstances))
        grid = [0] * row * row
        xy = jnp.indices([row, row], dtype=jnp.float32)
        offset = -(row - 1) / 2.
        x = ((xy[1].reshape(row * row) + offset) * 2).tolist()
        y = ((xy[0].reshape(row * row) + offset) * 2).tolist()
        data = [x, y, grid]
        await self.send(text_data=json.dumps(data))




    async def initGOL(self, nbInstances):
        key = jax.random.PRNGKey(1701)
        row = int(math.sqrt(nbInstances))
        self.grid = jax.random.uniform(key, (1, row, row, 1))
        self.grid = jnp.round(self.grid)
        self.grid = jnp.transpose(self.grid, [0, 3, 1, 2])

        # HWIO
        self.kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
        self.kernel += jnp.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])[:, :, jnp.newaxis, jnp.newaxis]
        self.kernel = jnp.transpose(self.kernel, [3, 2, 0, 1])
        xy = jnp.indices([row, row], dtype=jnp.float32)
        offset = -(row - 1) / 2.
        self.x = ((xy[1].reshape(row * row) + offset) * 2).tolist()
        self.y = ((xy[0].reshape(row * row) + offset) * 2).tolist()
        states = jnp.reshape(self.grid, (self.grid.size)).tolist()
        data = [self.x, self.y, states]
        await self.send(text_data=json.dumps(data))

    async def sendOneStepGOL(self):
        def update(grid, kernel):
            out = lax.conv(grid, kernel, (1, 1), 'SAME')

            cdt_1 = out == 12 
            cdt_2 = out == 13
            cdt_3 = out == 3

            out = jnp.logical_or(cdt_1, cdt_2)
            out = jnp.logical_or(out, cdt_3)
            return out.astype(jnp.float32)
        
        states = jnp.reshape(self.grid, (self.grid.size)).tolist()
        data = [self.x, self.y, states]
        await self.send(text_data=json.dumps(data))
        self.grid = update(self.grid, self.kernel)