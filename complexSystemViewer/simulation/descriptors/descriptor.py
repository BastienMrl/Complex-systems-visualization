from abc import ABC, abstractmethod

import jax.numpy as jnp

class Descriptor(ABC):
    def __init__():
        pass

    @abstractmethod
    def get_descriptor(grid : jnp.ndarray) -> jnp.ndarray:
        return None
    
    @abstractmethod
    def get_descriptor_infos() -> str | list[str]:
        return "Not implemented descriptor"