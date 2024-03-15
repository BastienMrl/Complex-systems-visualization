import jax.numpy as jnp
from .state import *
from typing import Callable

class Interaction():
    def __init__(self, id : str, apply_fct : Callable[[jnp.ndarray, list[State]], None]):
        self.id = id
        self.apply_fct = apply_fct

    def apply(self, mask : jnp.ndarray, states : list[State]):
        self.apply_fct(mask, states)
