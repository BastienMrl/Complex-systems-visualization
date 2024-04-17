import jax.numpy as np
import jax.lax as lax
import jax.random
from ..utils import Timer

from ..simulation import *


class PhysarumAgentParameters(SimulationParameters):

    def __init__(self, id_prefix : str = "default"):
        super().__init__(id_prefix)

        #init
        self.nb_agents : int
        self.grid_size : int

        #rules
        self.speed : float
        self.dist_sensor : float
        self.angle_sensor : float
        self.rotation_angle : float
        self.sensing_channel : int

        self._init_param : list[Param] = [
            IntParam(id_p= self._get_id_prefix() + "nbAgents", name= self._get_name_prefix() + "Nb of Agents", default_value= 10000,
                    min_value = 100, max_value= 60000, step=1000),
            IntParam(id_p= self._get_id_prefix() + "gridSize", name = self._get_name_prefix() + "Grid Size", default_value=100,
                     min_value = 20, max_value= 2000, step=1)
        ]

        self._rules_param : list[Param] = [
            FloatParam(id_p= self._get_id_prefix() + "speed", name= self._get_name_prefix() + "Speed", default_value=1.2,
                    min_value= 0.4, max_value= 50, step = 0.1),
            FloatParam(id_p= self._get_id_prefix() + "distSensor", name= self._get_name_prefix() + "Sensor distance", default_value=8.,
                    min_value= 0.5, max_value=50, step= 0.1),
            FloatParam(id_p= self._get_id_prefix() + "angleSensor", name= self._get_name_prefix() + "Sensor angle", default_value=20,
                    min_value= 2, max_value=270, step=1),
            FloatParam(id_p= self._get_id_prefix() + "rotationAngle", name= self._get_name_prefix() + "Rotation", default_value=10,
                    min_value=2, max_value=170, step=1),
            IntParam(id_p= self._get_id_prefix() + "sensingChannel", name= self._get_name_prefix() + "Sensing Channel", default_value=1, min_value = 0, max_value = 10, step=1)
        ]

        self.set_all_params()

    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match (idx):
            case 0:
                self.nb_agents = param.value
            case 1:
                self.grid_size = param.value

    def rule_param_value_changed(self, idx: int, param: Param) -> None:
        match (idx):
            case 0: 
                self.speed = param.value
            case 1: 
                self.dist_sensor = param.value
            case 2:
                self.angle_sensor = param.value * jnp.pi / 180
            case 3:
                self.rotation_angle = param.value * jnp.pi / 180
            case 4:
                self.sensing_channel = param.value


class PhysarumAgentSimulation(Simulation):

    


    def __init__(self, params : PhysarumAgentParameters = PhysarumAgentParameters(), needJSON : bool = True):
        super().__init__(needJSON=needJSON)
        self.initSimulation(params)

    def initSimulation(self, params : PhysarumAgentParameters = PhysarumAgentParameters()):
        self.params : PhysarumAgentParameters = params

        key = jax.random.key(918)
        self.grid : jnp.ndarray = jax.random.uniform(key, (self.params.grid_size, self.params.grid_size, 1))

        self.current_states : ParticleState = None
        self.init_default_sim()

        if(self.NEED_JSON):
            self.to_JSON_object()

    def init_default_sim(self):
        self.init_random_sim()

    def init_random_sim(self):
        self.key = jax.random.key(101)
        positions = jax.random.uniform(self.key, (self.nb_agents, 2), minval= 0, maxval=self.params.grid_size - 1)
        orientation = jax.random.uniform(self.key, (self.nb_agents, 1), minval=0., maxval= 2 * jnp.pi)

        self.current_states = ParticleState(self.params.grid_size, self.params.grid_size, [0.], [jnp.pi * 2], jnp.hstack((positions, orientation)))
        self.current_id = 0

    def _step(self):
        timer = Timer("Step agents")
        timer.start()

        state : ParticleState = self.current_states
        x = state.get_pos_x()
        y = state.get_pos_y()
        orientation = state.get_value(0)


        self.key, subkey = jax.random.split(self.key)
        random = jax.random.uniform(subkey, [self.nb_agents])
        random = jnp.where(random < 0.5, -1., 1.)

        self.current_states.particles = _get_new_states(x, y, self.grid, random, self.params)

        timer.stop()


    def set_grid(self, grid : jnp.ndarray):
        self.grid = grid
    
@jax.jit
def _get_new_states(x : jnp.ndarray, y : jnp.ndarray, grid : jnp.ndarray, random_values : jnp.ndarray, params : PhysarumAgentParameters):
    x_front = jnp.round(x + params.dist_sensor * jnp.cos(orientation)).astype(dtype=jnp.int16)
    x_front = jnp.where(x_front < 0., x_front + (params.grid_size - 1), x_front)
    x_front = jnp.where(x_front > params.grid_size - 1, x_front - (params.grid_size - 1), x_front)
    y_front = jnp.round(y + params.dist_sensor * jnp.cos(orientation)).astype(dtype=jnp.int16)
    y_front = jnp.where(y_front < 0., y_front + (params.grid_size - 1), y_front)
    y_front = jnp.where(y_front > params.grid_size - 1, y_front - (params.grid_size - 1), y_front)

    x_left = jnp.round(x + params.dist_sensor * jnp.cos(orientation + params.angle_sensor)).astype(dtype=jnp.int16)
    x_left = jnp.where(x_left < 0., x_left + (params.grid_size - 1), x_left)
    x_left = jnp.where(x_left > params.grid_size - 1, x_left - (params.grid_size - 1), x_left)
    y_left = jnp.round(y + params.dist_sensor * jnp.sin(orientation + params.angle_sensor)).astype(dtype=jnp.int16)
    y_left = jnp.where(y_left < 0., y_left + (params.grid_size - 1), y_left)
    y_left = jnp.where(y_left > params.grid_size - 1, y_left - (params.grid_size - 1), y_left)

    x_right = jnp.round(x + params.dist_sensor * jnp.cos(orientation - params.angle_sensor)).astype(dtype=jnp.int16)
    x_right = jnp.where(x_right < 0., x_right + (params.grid_size - 1), x_right)
    x_right = jnp.where(x_right > params.grid_size - 1, x_right - (params.grid_size - 1), x_right)
    y_right = jnp.round(y + params.dist_sensor * jnp.sin(orientation - params.angle_sensor)).astype(dtype=jnp.int16)
    y_right = jnp.where(y_right < 0., y_right + (params.grid_size - 1), y_right)
    y_right = jnp.where(y_right > params.grid_size - 1, y_right - (params.grid_size - 1), y_right)


    left_value = grid[x_left, y_left, params.sensing_channel]
    front_value = grid[x_front, y_front, params.sensing_channel]
    right_value = grid[x_right, y_right, params.sensing_channel]

    left_greater_front = jnp.greater(left_value, front_value)
    front_greater_right = jnp.greater(front_value, right_value)

    cdt_left = jnp.logical_and(left_greater_front, front_greater_right)
    cdt_right = jnp.logical_and(jnp.logical_not(left_greater_front), jnp.logical_not(front_greater_right))
    # cdt_none = jnp.logical_not(jnp.logical_or(cdt_left, jnp.logical_or(cdt_right, cdt_front)))
    # cdt_none = jnp.logical_or(cdt_none, jnp.logical_and(jnp.equal(left_value, front_value), jnp.equal(right_value, front_value)))
    cdt_none = jnp.logical_and(jnp.equal(left_value, front_value), jnp.equal(right_value, front_value))

    choice = jnp.where(cdt_left, 1, 0)
    choice = jnp.where(cdt_right, -1, choice)
    choice = jnp.where(cdt_none, random_values, choice)

    orientation += choice * params.rotation_angle


    x += params.speed * jnp.cos(orientation)
    x = jnp.where(x < 0., x + (params.grid_size - 1), x)
    x = jnp.where(x > params.grid_size - 1, x - (params.grid_size - 1), x)
    y += params.speed * jnp.sin(orientation)
    y = jnp.where(y < 0., y + (params.grid_size - 1), y)
    y = jnp.where(y > params.grid_size - 1, y - (params.grid_size - 1), y)

    x = jnp.expand_dims(x, (1))    
    y = jnp.expand_dims(y, (1))
    orientation = jnp.expand_dims(orientation, (1))
    orientation = (orientation % (2 * jnp.pi))


    return jnp.hstack((x, y, orientation))


