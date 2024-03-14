import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import jax.scipy as jsp
import numpy as np
import typing as t
from functools import partial
import math
import chex


from ..simulation import * 
class LeniaSimulation(Simulation):

    initialization_parameters = [
        BoolParam(id_p="randomStart", name="Random start", default_value=False),
        IntParam(id_p="seed", name="Seed", default_value=6, min_value=1),
        IntParam(id_p="gridSize", name="Grid size",
                 default_value=200, min_value=40, step=1),
        FloatParam(id_p="dt", name="dt",
                    default_value=0.05, min_value=0.1, max_value=1., step=0.1),
        IntParam(id_p="C", name="Number of Channels",
                    default_value=1, min_value=1, max_value=4, step=1)
    ]

    default_rules = [
            IntParam(id_p="number_of_kernels", name="Number of kernels",
                    default_value=10, min_value=1, step=1),
            IntParam(id_p="dd", name="dd",
                    default_value=5, min_value=1, step=1),
            IntParam(id_p="n", name="n",
                    default_value=2, min_value=1, step=1),
            FloatParam(id_p="theta_A", name="Theta A",
                    default_value=2.0, min_value=0.0, max_value=2.0, step=0.1),
            FloatParam(id_p="sigma", name="Sigma",
                    default_value=0.65, min_value=0.0, max_value=1.0, step=0.05)]
    
    
    def __init__(self, init_states = None, rules = default_rules): 
        super().__init__()
        self.initSimulation(init_states, rules)
           
    def initSimulation(self, init_states : list[State] = None, rules : list[Param] = default_rules, init_param : list[Param]= initialization_parameters):
        
        self.random_start : bool = [p for p in init_param if p.id_param == "randomStart"][0].value
        self.grid_size : int = [p for p in init_param if p.id_param == "gridSize"][0].value
        self.dt : float = [p for p in init_param if p.id_param == "dt"][0].value
        self.C : int = [p for p in init_param if p.id_param == "C"][0].value


        
        self.rules = rules
        self.nb_k = self.getRuleById("number_of_kernels")
        SX = SY = self.grid_size
        
        self.M = np.ones((self.C, self.C), dtype=int) * self.nb_k
        self.nb_k = int(self.M.sum())
        self.c0, self.c1 = conn_from_matrix( self.M )
    
        self.rule_space = RuleSpace(self.nb_k)
        self.kernel_computer = KernelComputer(SX, SY, self.nb_k)

        seed : int = [p for p in init_param if p.id_param == "seed"][0].value
        
        key = jax.random.PRNGKey(seed)
        params_seed, state_seed = jax.random.split(key)
        params = self.rule_space.sample(params_seed)
        self.c_params = self.kernel_computer(params)

        if init_states != None:
            self.current_states = init_states
            self.width = init_states[0].width
            self.height = init_states[0].height
        elif self.random_start:
            self.init_random_sim(state_seed)
        else:
            self.init_default_sim()

        def lenia_interaction(mask : jnp.ndarray, states : list[State]):
            mask = jnp.expand_dims(mask, 2)
            shape = list(states[0].grid.shape)
            shape[-1] -= 1

            minus_one = jnp.subtract(jnp.zeros(shape), jnp.ones(shape))
            
            mask = jnp.dstack((mask, minus_one))


            states[0].grid = jnp.where(mask >= 0, mask, states[0].grid)

        self.interactions : list[Interaction] = [Interaction("toLife", lenia_interaction)]


    def set_current_state_from_array(self, new_state):
        state = new_state[2:]
        grid = jnp.asarray(state, dtype=jnp.float32).reshape((self.current_states[0].grid.shape))
        self.current_states[0].set_grid(grid)



    def init_default_sim(self):
        grid = jnp.zeros((self.grid_size, self.grid_size, self.C))
        state = GridState(grid)
        self.current_states = [state]
        self.width = self.grid_size
        self.height = self.grid_size

    def init_random_sim(self, state_seed):
        SX = SY = self.grid_size
        mx, my = SX//2, SY//2 # center coordinated
        offsetX, offsetY= round(SX/8), round(SY/8)


        A0 = jnp.zeros((SX, SY, self.C)).at[mx-offsetX:mx+offsetX, my-offsetY:my+offsetY, :].set(
            jax.random.uniform(state_seed, (2*offsetX, 2*offsetY, self.C))
        )
        state = GridState(A0)
        self.current_states = [state]
        self.width = self.grid_size
        self.height = self.grid_size

        
        
       

    def step(self) :

        A = self.current_states[0].grid

        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(self.c_params.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, self.c_params.m, self.c_params.s) * self.c_params.h  # (x,y,k)

        U = jnp.dstack([ U[:, :, self.c1[c]].sum(axis=-1) for c in range(self.C) ])  # (x,y,c)

        nA = jnp.clip(A + self.dt * U, 0., 1.)
        self.current_states[0].grid = nA
        
    
    





def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

ker_f = lambda x, a, w, b : (b * jnp.exp( - (x[..., None] - a)**2 / w)).sum(-1)

bell = lambda x, m, s: jnp.exp(-((x-m)/s)**2 / 2)

def growth(U, m, s):
    return bell(U, m, s)*2-1

kx = jnp.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
])
ky = jnp.transpose(kx)
def sobel_x(A):
    """
    A : (x, y, c)
    ret : (x, y, c)
    """
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], kx, mode = 'same') 
                    for c in range(A.shape[-1])])
def sobel_y(A):
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], ky, mode = 'same') 
                    for c in range(A.shape[-1])])
  
@jax.jit
def sobel(A):
    return jnp.concatenate((sobel_y(A)[:, :, None, :], sobel_x(A)[:, :, None, :]),
                            axis = 2)



def get_kernels(SX: int, SY: int, nb_k: int, params):
    mid = SX//2
    Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
          ((params['R']+15) * params['r'][k]) for k in range(nb_k) ]  # (x,y,k)
    K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params["a"][k], params["w"][k], params["b"][k]) 
                    for k, D in zip(range(nb_k), Ds)])
    nK = K / jnp.sum(K, axis=(0,1), keepdims=True)
    return nK


def conn_from_matrix(mat):
    C = mat.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = mat[s, t]
            if n:
                c0 = c0 + [s]*n
                c1[t] = c1[t] + list(range(i, i+n))
            i+=n
    return c0, c1


def conn_from_lists(c0, c1, C):
    return c0, [[i == c1[i] for i in range(len(c0))] for c in range(C)]
    

@chex.dataclass
class Params:
    """Flow Lenia update rule parameters
    """
    r: jnp.ndarray
    b: jnp.ndarray
    w: jnp.ndarray
    a: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray
    R: float


@chex.dataclass
class CompiledParams:
    """Flow Lenia compiled parameters
    """
    fK: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray



class RuleSpace :

    """Rule space for Flow Lenia system
    
    Attributes:
        kernel_keys (TYPE): Description
        nb_k (int): number of kernels of the system
        spaces (TYPE): Description
    """
    
    #-----------------------------------------------------------------------------
    def __init__(self, nb_k: int):
        """
        Args:
            nb_k (int): number of kernels in the update rule
        """
        self.nb_k = nb_k    
        self.kernel_keys = 'r b w a m s h'.split()
        self.spaces = {
            "r" : {'low' : .2, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            "b" : {'low' : .001, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "w" : {'low' : .01, 'high' : .5, 'mut_std' : .2, 'shape' : (3,)},
            "a" : {'low' : .0, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "m" : {'low' : .05, 'high' : .5, 'mut_std' : .2, 'shape' : None},
            "s" : {'low' : .001, 'high' : .18, 'mut_std' : .01, 'shape' : None},
            "h" : {'low' : .01, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            'R' : {'low' : 2., 'high' : 25., 'mut_std' : .2, 'shape' : None},
        }
    #-----------------------------------------------------------------------------
    def sample(self, key: jnp.ndarray)->Params:
        """sample a random set of parameters
        
        Returns:
            Params: sampled parameters
        
        Args:
            key (jnp.ndarray): random generation key
        """
        kernels = {}
        for k in 'rmsh':
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
              key=subkey, minval=self.spaces[k]['low'], maxval=self.spaces[k]['high'], 
              shape=(self.nb_k,)
            )
        for k in "awb":
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
              key=subkey, minval=self.spaces[k]['low'], maxval=self.spaces[k]['high'], 
              shape=(self.nb_k, 3)
            )
        R = jax.random.uniform(key=key, minval=self.spaces['R']['low'], maxval=self.spaces['R']['high'])
        return Params(R=R, **kernels)

class KernelComputer:

    """Summary
    
    Attributes:
        apply (Callable): main function transforming raw params (Params) in copmiled ones (CompiledParams)
        SX (int): X size
        SY (int): Y size
    """
    
    def __init__(self, SX: int, SY: int, nb_k: int):
        """Summary
        
        Args:
            SX (int): Description
            SY (int): Description
            nb_k (int): Description
        """
        self.SX = SX
        self.SY = SY

        mid = SX // 2
        def compute_kernels(params: Params)->CompiledParams:
            """Compute kernels and return a dic containing kernels fft
            
            Args:
                params (Params): raw params of the system
            
            Returns:
                CompiledParams: compiled params which can be used as update rule
            """

            Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
                  ((params.R+15) * params.r[k]) for k in range(nb_k) ]  # (x,y,k)
            K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, params.a[k], params.w[k], params.b[k]) 
                            for k, D in zip(range(nb_k), Ds)])
            nK = K / jnp.sum(K, axis=(0,1), keepdims=True)  # Normalize kernels 
            fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))  # Get kernels fft

            return CompiledParams(fK=fK, m=params.m, s=params.s, h=params.h)

        self.apply = jax.jit(compute_kernels)

    def __call__(self, params: Params):
        """callback to apply
        """
        return self.apply(params)
