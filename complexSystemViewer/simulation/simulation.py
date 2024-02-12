from abc import ABC, abstractmethod 
import jax.numpy as jnp
import jax.lax as lax
import jax.random
class Simulation(ABC): 
    
    s_id = None
    current_states = None
    parameters = None
    width = None
    height = None
     

    def __init__(self, id :int, init_states = None, init_params : ParamList = None): 
        s_id = None
        if init_states != None :
            self.current_states = init_states
        else : 
            raise ValueError("Initial State can't be None")
        
        self.width = init_states[0].width
        self.height = init_states[0].height
        for state in init_states : 
            if not jnp.isclose(state.width, self.width) or not jnp.isclose(state.height, self.height):
                raise ValueError("States of a simulation must be of same size")
        

        if init_params != None :
            self.parameter = init_params
        else :
            raise ValueError("Initial parameters can't be None")
    
    @abstractmethod
    def step(self) : 
        pass

    def to_JSON_object(self) :
      
        json_particles = list()
        id_arr = jnp.empty( shape=(0) )
        pos_arr = jnp.empty( shape=(0) )
        values_arr = jnp.empty( shape=(0) )
        class_arr = jnp.empty( shape=(0) )
        for state in current_states :
            particules = state.particles

            id_arr = jnp.append(
                jnp.fromiter((p.p_id for p in particules), dtype=int)
            )

            pos_arr = jnp.append(
                jnp.fromiter(((p.pos_x, p.pos_y) for p in particules), dtype=tuple)
            )

            values_arr = jnp.append(
                jnp.fromiter((p.values for p in particules), dtype=tuple)
            )

            class_arr = jnp.append(
                jnp.fromiter((p.particle_class for p in particules), dtype=int)
            )
            
        return {
            ids : id_arr.tolist(),
            positions : pos_arr.tolist(),
            values : values_arr.tolist(),
            classes : class_arr.tolist()  
        }
    