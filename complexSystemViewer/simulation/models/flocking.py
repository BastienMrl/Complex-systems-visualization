import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from jax import jit
from jax import vmap

from jax.config import config

from jax_md import space,energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

vectorize = np.vectorize

from functools import partial

from ..simulation import * 
from ..state import ParticleState
from dataclasses import dataclass
from collections import namedtuple

Boid = namedtuple('Boid', ['R', 'theta'])



class FlockingParameters(SimulationParameters):

    def __init__(self, id_prefix : str = "default"):
        super().__init__(id_prefix)
        
        #init
        self.box_size : int
        self.boid_count : int
        self.dt : float

        #rules
        self.speed : float
        self.d_align : float
        self.j_align : float

        self.d_avoid : float
        self.j_avoid : float
        
        self.d_cohesion : float
        self.j_cohesion : float

        #front
        self._rules_param : list[Param] = [
            FloatParam(id_p= self._get_id_prefix() + "speed", name= self._get_name_prefix() + "Boids speed",
                        default_value=10.0, min_value=0.0, max_value=40.0, step=0.1),
            FloatParam(id_p= self._get_id_prefix() + "D-align", name= self._get_name_prefix() + "Alignement Distance (D)",
                        default_value=45., min_value=0.0, max_value=100., step=5.),
            FloatParam(id_p= self._get_id_prefix() + "J_align", name= self._get_name_prefix() + "Alignement Strenght (J)",
                        default_value=1.0, min_value=0.0, max_value=10.0, step=0.1),
            FloatParam(id_p= self._get_id_prefix() + "D-avoid", name= self._get_name_prefix() + "Avoidance Distance (D)",
                        default_value=30., min_value=0.0, max_value=100., step=5.),
            FloatParam(id_p= self._get_id_prefix() + "J-avoid", name= self._get_name_prefix() + "Avoidance Strenght (J)",
                        default_value=25.0, min_value=0.0, max_value=100.0, step=1.),
            FloatParam(id_p= self._get_id_prefix() + "D-cohesion", name= self._get_name_prefix() + "Cohesion Distance (D)",
                        default_value=40.0, min_value=0.0, max_value=100.0, step=5.),
            FloatParam(id_p= self._get_id_prefix() + "J-cohesion", name= self._get_name_prefix() + "Cohesion Strenght (J)",
                        default_value=0.005, min_value=0.0, max_value=1., step=0.001),
        ]

        self._init_param : list[Param] = [   
            IntParam(id_p= self._get_id_prefix() + "boxSize", name= self._get_name_prefix() + "Box size",
                    default_value=400, min_value=10, step=10),
            IntParam(id_p= self._get_id_prefix() + "boidCount", name= self._get_name_prefix() + "Boid count",
                    default_value=300, min_value=1, step=5),
            FloatParam(id_p= self._get_id_prefix() + "dt", name= self._get_name_prefix() + "dt",
                        default_value=0.3, min_value=0.01, max_value=2., step=0.05),
        ]

        self.set_all_params()

    def init_param_value_changed(self, idx: int, param: Param) -> None:
        match (idx):
            case 0:
                self.box_size = param.value
            case 1:
                self.boid_count = param.value
            case 2:
                self.dt = param.value

    def rule_param_value_changed(self, idx: int, param : Param) -> None:
        match(idx):
            case 0:
                self.speed = param.value
            case 1:
                self.d_align = param.value
            case 2:
                self.j_align = param.value
            case 3:
                self.d_avoid = param.value
            case 4: 
                self.j_avoid = param.value
            case 5:
                self.d_cohesion = param.value
            case 6: 
                self.j_cohesion = param.value





class FlockingSimulation(Simulation): 

        
    def __init__(self, params : FlockingParameters = FlockingParameters()): 
        super().__init__()
        self.initSimulation(params)

    #methods added to simplify usage of jax
    def JAX_to_ParticleState(self, state) :
        boids = state['boids']

        theta = (boids.theta % (2 * jnp.pi)) / (2. * jnp.pi)

        particles = jnp.stack((boids.R[:, 0] - self.box_size / 2, boids.R[:, 1] - self.box_size / 2, theta), 1)

        self.current_states.particles = particles
            
        
    def particleState_to_JAX(self, state : ParticleState) :
        boids = Boid(
            jnp.array(np.vstack(state.particles[:, :2])),
            jnp.array(state.particles[:, 2] * 2 * jnp.pi)
        )
        return {'boids' :boids}


    def initSimulation(self, params : FlockingParameters = FlockingParameters()):
        self.params : FlockingParameters = params

        self.init_default_sim()
            

        self.interactions : list[Interaction] = None

        self.displacement, self.shift = space.periodic(self.params.box_size)

        self.state = self.particleState_to_JAX(self.current_states)
        self.to_JSON_object()

 
    def _step(self) :
        speed = self.getRuleById("speed")
        state = self.state
        R, theta = state['boids']
        
        dstate = quantity.force(self.energy_fn)(state)
        dR, dtheta = dstate['boids']
        n = normal(state['boids'].theta)

        state['boids'] = Boid(self.shift(R, self.params.dt * (speed * n + dR)), 
                            theta + self.params.dt * dtheta)
        self.JAX_to_ParticleState(state)


    def set_current_state_from_array(self, new_state):
        self.current_states = [self.JAX_to_ParticleState(new_state)]


    def init_default_sim(self):
        #self.current_states = [ParticleState(self.box_size, self.box_size)] #empty state not supported yet
        self.init_random_sim() #default behaviour for now

    def init_random_sim(self):
        key = jax.random.key(1)
        positions = jax.random.uniform(key, (self.params.boid_count, 2), minval= -self.params.box_size / 2, maxval=self.params.box_size / 2)
        theta = jax.random.uniform(key, (self.params.boid_count, 1))
        
        
        self.current_states = ParticleState(self.params.box_size, self.params.box_size, [0.], [1.],
                jnp.hstack((positions, theta)))
        self.current_states.id = 0

    def energy_fn(self, state):
        
        boids = state['boids']
        E_align = partial(align_fn, J_align=self.params.j_align, D_align=self.params.d_align, alpha=3.)
        # Map the align energy over all pairs of boids. While both applications
        # of vmap map over the displacement matrix, each acts on only one normal.
        E_align = vmap(vmap(E_align, (0, None, 0)), (0, 0, None))

        E_avoid = partial(avoid_fn, J_avoid=self.params.j_avoid, D_avoid=self.params.d_avoid, alpha=3.)
        E_avoid = vmap(vmap(E_avoid))

        E_cohesion = partial(cohesion_fn, J_cohesion=self.params.j_cohesion, D_cohesion=self.params.d_cohesion)

        dR = space.map_product(self.displacement)(boids.R, boids.R)
        N = normal(boids.theta)

        return (0.5 * np.sum(E_align(dR, N, N) + E_avoid(dR)) + 
          np.sum(E_cohesion(dR, N)))
    

    # def to_JSON_object(self) :
    #     boids = self.state['boids']
    #     pos_row = jnp.transpose(boids.R)
    #     x_row = (pos_row[0] - (self.box_size/2)).tolist()
    #     y_row = (pos_row[1] - (self.box_size/2)).tolist()
    #     domain = [self.boid_count, 1]
    #     val_row = ((boids.theta % (2 * jnp.pi)) / (2 * jnp.pi)).tolist()
    #     l = [domain, x_row, y_row, val_row]
    #     self.as_json = l
    

@vmap
def normal(theta):
    return jnp.array([jnp.cos(theta), jnp.sin(theta)])

def align_fn(dR, N_1, N_2, J_align, D_align, alpha):
    dR = lax.stop_gradient(dR)
    dr = space.distance(dR) / D_align
    energy = J_align / alpha * (1. - dr) ** alpha * (1 - jnp.dot(N_1, N_2)) ** 2
    return jnp.where(dr < 1.0, energy, 0.)

def avoid_fn(dR, J_avoid, D_avoid, alpha):
  dr = space.distance(dR) / D_avoid
  return jnp.where(dr < 1., 
                  J_avoid / alpha * (1 - dr) ** alpha, 
                  0.)

def cohesion_fn(dR, N, J_cohesion, D_cohesion, eps=1e-7):
  dR = lax.stop_gradient(dR)
  dr = jnp.linalg.norm(dR, axis=-1, keepdims=True)
  
  mask = dr < D_cohesion

  N_com = jnp.where(mask, 1.0, 0)
  dR_com = jnp.where(mask, dR, 0)
  dR_com = jnp.sum(dR_com, axis=1) / (jnp.sum(N_com, axis=1) + eps)
  dR_com = dR_com / jnp.linalg.norm(dR_com + eps, axis=1, keepdims=True)
  return f32(0.5) * J_cohesion * (1 - jnp.sum(dR_com * N, axis=1)) ** 2



