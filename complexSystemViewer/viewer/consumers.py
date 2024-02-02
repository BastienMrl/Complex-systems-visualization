import json
import asyncio
import random
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
                self.simulation_task = asyncio.create_task(self.sendRandomBool(text_data_json["params"]))
        elif message == "Stop":
            print(message)
            if self.simulation_task:
                self.simulation_task.cancel()
                self.simulation_task = None

    async def sendRandomBool(self, nb):
        def generate_states():
            l = []
            for i in range(nb):
                l.append(bool(random.getrandbits(1)))
            return l
        
        try:
            while True:
                await self.send(text_data=json.dumps(generate_states()))
                await asyncio.sleep(0.01)  # Contrôle la vitesse de la simulation
        except asyncio.CancelledError:
            pass  # La tâche a été annulée, arrêter la simulation proprement