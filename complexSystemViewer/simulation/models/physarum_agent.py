import jax.numpy as np
import jax.lax as lax
import jax.random
from ..utils import Timer

from ..simulation import *

class PhysarumAgentSimulation(Simulation):

    initialization_parameters = [
        IntParam(id_p="nbAgents", name="Nb of Agents", default_value= 1,
                 min_value = 1, max_value= 10000, step=1),
        IntParam(id_p="gridSize", name = "Grid Size",
                 default_value=20, min_value = 10, step=10)
    ]

    default_rules = [
        FloatParam(id_p="speed", name="Speed", default_value=1.,
                   min_value= 1., max_value= 20, step = 0.5),
        FloatParam(id_p="distSensor", name="Sensor distance", default_value=1.,
                   min_value= 0.5, max_value=5, step= 0.1),
        FloatParam(id_p="angleSensor", name="Sensor angle", default_value=45,
                   min_value= 10, max_value=270, step=1),
        FloatParam(id_p="rotationAngle", name="Rotation", default_value=90,
                   min_value=10, max_value=170, step=1)
    ]


    def __init__(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters, needJSON : bool = True):
        super().__init__(needJSON=needJSON)
        self.initSimulation(rules, init_param)

    def initSimulation(self, rules : list[Param] = default_rules, init_param : list[Param] = initialization_parameters):
        self.rules = rules
        self.init_param = init_param

        self.nb_agents = self.get_init_param("nbAgents").value
        self.grid_size = self.get_init_param("gridSize").value

        self.speed = self.get_rules_param("speed").value
        self.dist_sensor = self.get_rules_param("distSensor").value
        self.angle_sensor = self.get_rules_param("angleSensor").value * jnp.pi / 180
        self.rotation_angle = self.get_rules_param("rotationAngle").value * jnp.pi / 180

        key = jax.random.key(918)
        self.grid : jnp.ndarray = jax.random.uniform(key, (self.grid_size, self.grid_size))


        self.current_states : ParticleState = None
        self.init_default_sim()

        if(self.NEED_JSON):
            self.to_JSON_object()

    def updateRule(self, json):
        super().updateRule(json)
        self.speed = self.get_rules_param("speed").value
        self.dist_sensor = self.get_rules_param("distSensor").value
        self.angle_sensor = self.get_rules_param("angleSensor").value * jnp.pi / 180
        self.rotation_angle = self.get_rules_param("rotationAngle").value * jnp.pi / 180

    def init_default_sim(self):
        self.init_random_sim()

    def init_random_sim(self):
        self.key = jax.random.key(101)
        positions = jax.random.uniform(self.key, (self.nb_agents, 2), minval= 0, maxval=self.grid_size - 1)
        orientation = jax.random.uniform(self.key, (self.nb_agents, 1), minval=0., maxval= 2 * jnp.pi)

        self.current_states = ParticleState(self.grid_size, self.grid_size, [0.], [jnp.pi * 2], jnp.hstack((positions, orientation)))
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


        self.current_states.particles = _get_new_states(x, y, self.grid, random, self.dist_sensor, self.angle_sensor, orientation,
                                                             self.grid_size, self.speed, self.rotation_angle)

        timer.stop()


    def set_grid(self, grid : jnp.ndarray):
        self.grid = grid

    def get_rules() -> list[Param] | None:
        return PhysarumAgentSimulation.default_rules

    def get_initialization() -> list[Param] | None:
        return PhysarumAgentSimulation.initialization_parameters
    
@jax.jit
def _get_new_states(x : jnp.ndarray, y : jnp.ndarray, grid : jnp.ndarray, random_values : jnp.ndarray,
                        dist_sensor : float, angle_sensor : float, orientation : jnp.ndarray, grid_offset : int, speed : float, rotation_angle : float):
    x_front = jnp.round(x + dist_sensor * jnp.cos(orientation)).astype(dtype=jnp.int16)
    x_front = jnp.where(x_front < 0., x_front + grid_offset, x_front)
    x_front = jnp.where(x_front > grid_offset - 1, x_front - (grid_offset - 1), x_front)
    y_front = jnp.round(y + dist_sensor * jnp.cos(orientation)).astype(dtype=jnp.int16)
    y_front = jnp.where(y_front < 0., y_front + grid_offset, y_front)
    y_front = jnp.where(y_front > grid_offset - 1, y_front - (grid_offset - 1), y_front)

    x_left = jnp.round(x + dist_sensor * jnp.cos(orientation + angle_sensor)).astype(dtype=jnp.int16)
    x_left = jnp.where(x_left < 0., x_left + grid_offset, x_left)
    x_left = jnp.where(x_left > grid_offset - 1, x_left - (grid_offset - 1), x_left)
    y_left = jnp.round(y + dist_sensor * jnp.sin(orientation + angle_sensor)).astype(dtype=jnp.int16)
    y_left = jnp.where(y_left < 0., y_left + grid_offset, y_left)
    y_left = jnp.where(y_left > grid_offset - 1, y_left - (grid_offset - 1), y_left)

    x_right = jnp.round(x + dist_sensor * jnp.cos(orientation - angle_sensor)).astype(dtype=jnp.int16)
    x_right = jnp.where(x_right < 0., x_right + grid_offset, x_right)
    x_right = jnp.where(x_right > grid_offset - 1, x_right - (grid_offset - 1), x_right)
    y_right = jnp.round(y + dist_sensor * jnp.sin(orientation - angle_sensor)).astype(dtype=jnp.int16)
    y_right = jnp.where(y_right < 0., y_right + grid_offset, y_right)
    y_right = jnp.where(y_right > grid_offset - 1, y_right - (grid_offset - 1), y_right)


    left_value = grid[x_left, y_left].squeeze(1)
    front_value = grid[x_front, y_front].squeeze(1)
    right_value = grid[x_right, y_right].squeeze(1)

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

    orientation += choice * rotation_angle


    x += speed * jnp.cos(orientation)
    x = jnp.where(x < 0., x + grid_offset, x)
    x = jnp.where(x > grid_offset - 1, x - (grid_offset - 1), x)
    y += speed * jnp.sin(orientation)
    y = jnp.where(y < 0., y + grid_offset, y)
    y = jnp.where(y > grid_offset - 1, y - (grid_offset - 1), y)

    x = jnp.expand_dims(x, (1))    
    y = jnp.expand_dims(y, (1))
    orientation = jnp.expand_dims(orientation, (1))
    orientation = (orientation % (2 * jnp.pi))


    return jnp.hstack((x, y, orientation))


