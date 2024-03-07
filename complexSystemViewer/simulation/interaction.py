import jax.numpy as jnp
from .state import *
from typing import Callable

class Interaction():
    def __init__(self, id : str, apply_fct : Callable[[jnp.ndarray, list[State]], None]):
        self.id = id
        self.apply_fct = apply_fct

    def apply(self, mask : jnp.ndarray, states : list[State]):
        self.apply_fct(mask, states)


def golInteractionReplacement(mask : jnp.ndarray, states : list[State]):
        to_zero = jnp.logical_not(jnp.logical_and(mask >= 0, mask < 0.5))
        to_one = jnp.logical_and(mask >= 0, mask >= 0.5)
        states[0].grid = jnp.logical_or(states[0].grid, to_one).astype(jnp.float32)
        states[0].grid = jnp.logical_and(states[0].grid, to_zero).astype(jnp.float32)

def leniaInteraction(mask : jnp.ndarray, states : list[State]):
    input = mask >= 0.
    if (len(states[0].grid.shape) == 3):
        input = jnp.expand_dims(input, 2)
        states[0].grid = jnp.where(input, mask, states[0].grid)