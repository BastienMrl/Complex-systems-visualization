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
                    await self.send_one_step()
            case "ResetSimulation":
                if self.isConnected:
                    await self.reset_simulation()
            case "ChangeSimulation":
                if self.isConnected:
                    self.sim = None
                    await self.init_new_simulation(text_data_json["simuName"])
            case "ResetRandomSimulation":
                if self.isConnected:
                    await self.reset_random_simulation()       
            case "UpdateInitParams":
                    await self.update_init_params(text_data_json["params"])
            case "UpdateRule":
                if self.isConnected:
                    await self.update_rule(text_data_json["params"])
            case "ApplyInteraction":
                if self.isConnected:
                    await self.apply_interaction(text_data_json["mask"], text_data_json["id"], text_data_json["interaction"])

    async def send_one_step(self):
        await self.send(bytes_data=orjson.dumps(self.sim.as_json))
        self.sim.new_step()

    async def update_init_params(self, params):
        json = orjson.loads(params)
        self.next_params_set.update_init_param(json)

    async def update_rule(self, params):
        self.sim.params.update_rules_param(orjson.loads(params))

    async def init_new_simulation(self, name):
        self.params = SimulationManager.get_parameters(name)
        self.next_params_set = SimulationManager.get_parameters(name)
        self.sim = SimulationManager.get_simulation_model(name, self.params)
        await self.send_one_step()    

    async def reset_simulation(self):
        self.params.set_init_from_list(self.next_params_set.get_init_parameters())
        self.next_params_set = cp.deepcopy(self.next_params_set)
        self.sim.init_simulation(self.params)
        await self.send_one_step()

    async def reset_random_simulation(self):
        self.params.set_init_from_list(self.next_params_set.get_init_parameters())
        self.next_params_set = cp.deepcopy(self.next_params_set)
        self.sim.init_random_simulation(self.params)
        await self.send_one_step()

    async def apply_interaction(self, mask : list[float], stateId : int, interaction : str):        
        self.sim.set_current_state_from_id(stateId)
        mask_width = int(math.sqrt(len(mask)))
        
        mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
        mask_jnp = mask_jnp.reshape((mask_width, mask_width))
        mask_jnp = jimage.resize(mask_jnp, (self.sim.width, self.sim.height), "linear")
        self.sim.apply_interaction(interaction, mask_jnp)
        await self.send_one_step()        

    
