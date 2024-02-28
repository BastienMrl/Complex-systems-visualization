import random
import jax.numpy as jnp
import jax.lax as lax
import jax.random
import jax.scipy as jsp
import numpy as np
import typing as t
from functools import partial
import matplotlib.pyplot as plt
import math
import chex


from ..simulation import * 
class LeniaSimulation(Simulation): 
    lenia = None
    c_param=None
    name = "Lenia"
    #test 
    default_parameters = [
            IntParam(id_p="number_of_kernels", name="Number of kernels",
                    default_value=10, min_value=1, step=1),
            IntParam(id_p="C", name="C",
                    default_value=1, min_value=1, step=1),
            IntParam(id_p="dd", name="dd",
                    default_value=5, min_value=1, step=1),
            IntParam(id_p="n", name="n",
                    default_value=2, min_value=1, step=1),
            FloatParam(id_p="dt", name="dt",
                    default_value=0.2, min_value=0.1, max_value=1., step=0.1),
            FloatParam(id_p="theta_A", name="Theta A",
                    default_value=2.0, min_value=0.0, max_value=2.0, step=0.1),
            FloatParam(id_p="sigma", name="Sigma",
                    default_value=0.65, min_value=0.0, max_value=1.0, step=0.05),
            IntParam(id_p="gridSize", name="Grid size",
                    default_value=50, min_value=0, step=1),
            RangeIntParam(id_p="birth", name="Birth",
                        min_param= IntParam(
                            id_p="",
                            name="",
                            default_value=3,
                            min_value=0,
                            max_value=8,
                            step=1
                        ),
                        max_param= IntParam(
                            id_p="",
                            name="",
                            default_value=3,
                            min_value=0,
                            max_value=8,
                            step=1
                        )),
            RangeIntParam(id_p="survival", name="Survival",
                        min_param= IntParam(
                            id_p="",
                            name="",
                            default_value=2,
                            min_value=0,
                            max_value=8,
                            step=1
                        ),
                        max_param= IntParam(
                            id_p="",
                            name="",
                            default_value=3,
                            min_value=0,
                            max_value=8,
                            step=1
                      ))]
    def __init__(self, init_states = None, init_params = default_parameters): 
        super().__init__(0, init_states, init_params)
        
        self.nb_k = self.getParamById("number_of_kernels")
        SX = SY = self.getParamById("gridSize")
        
        self.M = np.ones((self.getParamById("C"), self.getParamById("C")), dtype=int) * self.nb_k
        self.nb_k = int(self.M.sum())
        self.c0, self.c1 = conn_from_matrix( self.M )
        
        #config = Config(SX=SX, SY=SY, nb_k=nb_k, C=C, c0=c0, c1=c1, 
        #                dt=dt, theta_A=theta_A, dd=5, sigma=sigma)

        


        self.rule_space = RuleSpace(self.nb_k)

        self.kernel_computer = KernelComputer(SX, SY, self.nb_k)

        seed = 10
        key = jax.random.PRNGKey(seed)
        params_seed, state_seed = jax.random.split(key)
        params = self.rule_space.sample(params_seed)
        self.c_params = self.kernel_computer(params)

        self.RT = ReintegrationTracking(SX, SY, self.getParamById('dt'), 
            self.getParamById('dd'), self.getParamById('sigma'), "wall")#TODO 

        

        
        
       

    def step(self) :

        A = self.current_states[0].grid

        fA = jnp.fft.fft2(A, axes=(0,1))  # (x,y,c)

        fAk = fA[:, :, self.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(self.c_params.fK * fAk, axes=(0,1)))  # (x,y,k)

        U = growth(U, self.c_params.m, self.c_params.s) * self.c_params.h  # (x,y,k)

        U = jnp.dstack([ U[:, :, self.c1[c]].sum(axis=-1) for c in range(self.getParamById('C')) ])  # (x,y,c)

        nA = jnp.clip(A + self.getParamById("dt") * U, 0., 1.)
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

class ReintegrationTracking:

    def __init__(self, SX=256, SY=256, dt=.2, dd=5, sigma=.65, border="wall", has_hidden=False, 
                 hidden_dims=None, mix="softmax"):
        self.SX = SX
        self.SY = SY
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.has_hidden = has_hidden
        self.hidden_dims = hidden_dims
        self.border = border if border in ['wall', 'torus'] else 'wall'
        self.mix = mix
        
        self.apply = self._build_apply()

    def __call__(self, *args):
        return self.apply(*args)

    def _build_apply(self):

        x, y = jnp.arange(self.SX), jnp.arange(self.SY)
        X, Y = jnp.meshgrid(x, y)
        pos = jnp.dstack((Y, X)) + .5 #(SX, SY, 2)
        dxs = []
        dys = []
        dd = self.dd
        for dx in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                dxs.append(dx)
                dys.append(dy)
        dxs = jnp.array(dxs)
        dys = jnp.array(dys)
        #-----------------------------------------------------------------------------------------------
        if not self.has_hidden:

            @partial(jax.vmap, in_axes=(None, None, 0, 0))
            def step(X, mu, dx, dy):
                Xr = jnp.roll(X, (dx, dy), axis = (0, 1))
                mur = jnp.roll(mu, (dx, dy), axis = (0, 1))
                if self.border == 'torus':
                    dpmu = jnp.min(jnp.stack(
                        [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                        for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                    ), axis = 0)
                else :
                    dpmu = jnp.absolute(pos[..., None] - mur)
                sz = .5 - dpmu + self.sigma
                area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
                nX = Xr * area
                return nX
        
            def apply(X, F):

                ma = self.dd - self.sigma  # upper bound of the flow maggnitude
                mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
                if self.border == "wall":
                    mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)
                nX = step(X, mu, dxs, dys).sum(axis = 0)
                
                return nX
        #-----------------------------------------------------------------------------------------------
        else :



            @partial(jax.vmap, in_axes = (None, None, None, 0, 0))
            def step_flow(X, H, mu, dx, dy):
                """Summary
                """
                Xr = jnp.roll(X, (dx, dy), axis = (0, 1))
                Hr = jnp.roll(H, (dx, dy), axis = (0, 1)) #(x, y, k)
                mur = jnp.roll(mu, (dx, dy), axis = (0, 1))

                if self.border == 'torus':
                    dpmu = jnp.min(jnp.stack(
                        [jnp.absolute(pos[..., None] - (mur + jnp.array([di, dj])[None, None, :, None])) 
                        for di in (-self.SX, 0, self.SX) for dj in (-self.SY, 0, self.SY)]
                    ), axis = 0)
                else :
                    dpmu = jnp.absolute(pos[..., None] - mur)

                sz = .5 - dpmu + self.sigma
                area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)) , axis = 2) / (4 * self.sigma**2)
                nX = Xr * area
                return nX, Hr

            def apply(X, H, F):

                ma = self.dd - self.sigma  # upper bound of the flow maggnitude
                mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma) #(x, y, 2, c) : target positions (distribution centers)
                if self.border == "wall":
                    mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)
                nX, nH = step_flow(X, H, mu, dxs, dys)

                if self.mix == 'avg':
                    nH = jnp.sum(nH * nX.sum(axis = -1, keepdims = True), axis = 0)  
                    nX = jnp.sum(nH, axis = 0)
                    nH = nH / (nX.sum(axis = -1, keepdims = True)+1e-10)

                elif self.mix == "softmax":
                    expnX = jnp.exp(nX.sum(axis = -1, keepdims = True)) - 1
                    nX = jnp.sum(nX, axis = 0)
                    nH = jnp.sum(nH * expnX, axis = 0) / (expnX.sum(axis = 0)+1e-10) #avg rule

                elif self.mix == "stoch":
                    categorical=jax.random.categorical(
                      jax.random.PRNGKey(42), 
                      jnp.log(nX.sum(axis = -1, keepdims = True)), 
                      axis=0)
                    mask=jax.nn.one_hot(categorical,num_classes=(2*self.dd+1)**2,axis=-1)
                    mask=jnp.transpose(mask,(3,0,1,2)) 
                    nH = jnp.sum(nH * mask, axis = 0)
                    nX = jnp.sum(nX, axis = 0)

                elif self.mix == "stoch_gene_wise":
                    mask = jnp.concatenate(
                      [jax.nn.one_hot(jax.random.categorical(
                                                            jax.random.PRNGKey(42), 
                                                            jnp.log(nX.sum(axis = -1, keepdims = True)), 
                                                            axis=0),
                                      num_classes=(2*dd+1)**2,axis=-1)
                      for _ in range(self.hidden_dims)], 
                      axis = 2)
                    mask=jnp.transpose(mask,(3,0,1,2)) # (2dd+1**2, x, y, nb_k)
                    nH = jnp.sum(nH * mask, axis = 0)
                    nX = jnp.sum(nX, axis = 0)
                
                return nX, nH

        return apply
    

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
