import jax.numpy as jnp
from .state import *

class Interaction():
    def __init__(self, id : str):
        self.id = id

    def apply(self, mask : jnp.ndarray, states : list[State]):
        input = mask >= 0.1
        if (len(states[0].grid.shape) == 3):
            input = jnp.expand_dims(input, 2)
        states[0].grid = jnp.logical_or(states[0].grid, input).astype(jnp.float32)