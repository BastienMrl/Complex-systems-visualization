import json
import asyncio
import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import math
from channels.generic.websocket import AsyncWebsocketConsumer


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
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        if message == "Start":
            if self.simulation_task is None or self.simulation_task.done():
                key = jax.random.PRNGKey(1701)
                nb = text_data_json["params"]
                row = int(math.sqrt(nb))
                grid = jax.random.uniform(key, (1, row, row, 1))
                grid = jnp.round(grid)
                grid = jnp.transpose(grid, [0, 3, 1, 2])

                # HWIO
                kernel = jnp.zeros((3, 3, 1, 1), dtype=jnp.float32)
                kernel += jnp.array([[1, 1, 1],
                                    [1, 10, 1],
                                    [1, 1, 1]])[:, :, jnp.newaxis, jnp.newaxis]
                kernel = jnp.transpose(kernel, [3, 2, 0, 1])
                self.simulation_task = asyncio.create_task(self.sendGameOfLife(grid, kernel))
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


    async def sendGameOfLife(self, grid, kernel):

        def update(grid, kernel):
            out = lax.conv(grid, kernel, (1, 1), 'SAME')

            cdt_1 = out == 12 
            cdt_2 = out == 13
            cdt_3 = out == 3

            out = jnp.logical_or(cdt_1, cdt_2)
            out = jnp.logical_or(out, cdt_3)
            return out.astype(jnp.float32)
            
        try:
            while True:
                await self.send(text_data=json.dumps(jnp.reshape(grid, (grid.size)).tolist()))
                grid = update(grid, kernel)
                await asyncio.sleep(0.15)  # Contrôle la vitesse de la simulation
        except asyncio.CancelledError:
            pass  # La tâche a été annulée, arrêter la simulation proprement