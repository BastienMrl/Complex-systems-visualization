import random
import jax.numpy as jnp
import jax.lax as lax
import random as rand
import math
import numpy as np
from jax import random

from jax import jit
from jax import vmap

import os
from jax.config import config ; config.update('jax_enable_x64', True)
from jax_md import space,energy, minimize, quantity, simulate, partition, util
from jax_md.util import f32

vectorize = np.vectorize

from functools import partial

from ..simulation import * 
from ..state import ParticleState
from ..particle import Particle
from dataclasses import dataclass
from collections import namedtuple

Boid = namedtuple('Boid', ['R', 'theta'])

#BoidsParticle





class FlockingSimulation(Simulation): 

    initialization_parameters = [
        
        IntParam(id_p="boxSize", name="Box size",
                 default_value=100, min_value=10, step=10),
        IntParam(id_p="boidCount", name="Boid size",
                 default_value=200, min_value=1, step=5),
        FloatParam(id_p="dt", name="dt",
                    default_value=0.05, min_value=0.1, max_value=1., step=0.1),
    ]

    default_rules = [
        FloatParam(id_p="speed", name="Speed",
                    default_value=1.0, min_value=0.0, max_value=10.0, step=0.1),
        ]

    #methods added to simplify usage of jax
    def JAX_to_ParticleState(self, state) :
        boids = state['boids']
        
        for i in range(len(boids.theta)):
            p = self.current_states[0].particles[i]
            p.pos_x = boids.R[i][0]
            p.pos_y = boids.R[i][1]
            p.values = [boids.theta[i]/(2*np.pi)]
            
        
    def particleState_to_JAX(self, state) :
        v_R = vectorize(lambda p : jnp.array([p.pos_x, p.pos_y]), otypes=[jnp.ndarray])
        v_theta = vectorize(lambda p : p.values[0] * 2 * np.pi)
        boids = Boid(
            jnp.array(np.stack(v_R(state.particles))),
            jnp.array(v_theta(state.particles))
        )
        return {'boids' :boids}

    def __init__(self, init_states = None, rules = default_rules): 
        super().__init__()
        self.initSimulation(init_states, rules)

    def initSimulation(self, init_states = None, rules = default_rules, init_param = initialization_parameters):

        self.dt = [p for p in init_param if p.id_param == "dt"][0].value
        self.box_size = [p for p in init_param if p.id_param == "boxSize"][0].value
        self.boid_count = [p for p in init_param if p.id_param == "boidCount"][0].value

        self.rules = rules

        if init_states != None:
            self.current_states = init_states
        else:
            self.init_default_sim()
            

        self.interactions : list[Interaction] = None

        self.displacement, self.shift = space.periodic(self.box_size)

        self.state = self.particleState_to_JAX(self.current_states[0])


 
    def step(self) :
        speed = [p for p in self.rules if p.id_param == "speed"][0].value
        state =  self.state
        R, theta = state['boids']
        
        dstate = quantity.force(self.energy_fn)(state)
        dR, dtheta = dstate['boids']
        n = normal(state['boids'].theta)

        state['boids'] = Boid(self.shift(R, self.dt * (speed * n + dR)), 
                            theta + self.dt * dtheta)
        #self.JAX_to_ParticleState(state)
        self.state = state


    def set_current_state_from_array(self, new_state):
        self.current_states = [self.JAX_to_ParticleState(new_state)]


    def init_default_sim(self):
        #self.current_states = [ParticleState(self.box_size, self.box_size)] #empty state not supported yet
        self.init_random_sim() #default behaviour for now

    def init_random_sim(self):
        self.current_states = [ParticleState(self.box_size, self.box_size, 
                [
                    Particle(
                        random.random()*self.box_size,
                        random.random()*self.box_size, 
                        [random.random()]) 
                    for _ in range(self.boid_count)
                ])]
    def energy_fn(self, state):
        
        boids = state['boids']
        E_align = partial(align_fn, J_align=0.5, D_align=45., alpha=3.)
        # Map the align energy over all pairs of boids. While both applications
        # of vmap map over the displacement matrix, each acts on only one normal.
        E_align = vmap(vmap(E_align, (0, None, 0)), (0, 0, None))

        E_avoid = partial(avoid_fn, J_avoid=25., D_avoid=30., alpha=3.)
        E_avoid = vmap(vmap(E_avoid))

        dR = space.map_product(self.displacement)(boids.R, boids.R)
        N = normal(boids.theta)

        return 0.5 * np.sum(E_align(dR, N, N)  + E_avoid(dR))

    def to_JSON_object(self) :
        boids = self.state['boids']
        pos_row = jnp.transpose(boids.R)
        x_row = (pos_row[0] - (self.box_size/2)).tolist()
        y_row = (pos_row[1] - (self.box_size/2)).tolist()
        domain = [self.boid_count, 1]
        val_row = ((boids.theta % (jnp.pi)) / (jnp.pi)).tolist()
        l = [domain, x_row, y_row, val_row]
        return l

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





