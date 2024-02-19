class Particle(): 
    p_id : int = None
    pos_x : float = None
    pos_y : float = None
    values : tuple = None
    particle_class : int = None
    is_aligned_grid : bool = False

    def __init__(self, p_id, x,y,values : tuple,particle_class = 0, is_aligned_grid = False):
        self.p_id = p_id
        self.pos_x = x
        self.pos_y = y
        self.values = values
        self.particle_class = particle_class
        self.is_aligned_grid = is_aligned_grid

    """ def to_JSON_object(self) :
        return {
            "id" : self.p_id,
            "pos" : [self.pos_x,self.pos_y],
   
            "values" : self.values,
            "class" : self.particle_class
            #"is_aligned_grid" : self.is_aligned_grid
        } """



    