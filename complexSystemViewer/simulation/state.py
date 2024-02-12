from abc import ABC, abstractmethod 
class State(ABC): 
    width : int = None
    height : int = None

    particles : list = None

    def __init__(self, height : float, width : float, particles : list = None):
        if self.height > 0 :
            self.height = height
        else :
            raise ValueError("Height must be a postive float")
        if self.width > 0 :
            self.width = width
        else :
            raise ValueError("width must be a postive float")
        self.particles = particles
    
    def set_grid(self, grid) : 
        self.grid = grid
        
        