import sys
import matplotlib.pyplot as plt


FILE = sys.argv[1]


class Configuration() :
    def __init__(self, model : str, nb_element : int, nb_channel : int):
        self.model = model
        self.nb_element = nb_element
        self.nb_channel = nb_channel

    def get_config(self) -> dict:
        return {"model" : self.model,
                "nb_element" : self.nb_element,
                "nb_channel" : self.nb_channel}
    
    def is_valid(self) -> bool:
        value = self.model != "default"
        value &= self.nb_element != 0
        value &= self.nb_channel != 0
        return value

    

class SampleMean() :
    def __init__(self):
        self.rendering = 0.
        self.updating = 0.
        self.picking = 0.
        self.fps = 0.

        self.n_rendering = 0
        self.n_updating = 0
        self.n_picking = 0
        self.n_fps = 0

    def add_value(self, name : str, value : float):
        match (name):
            case "rendering" :
                self._add_rendering(value)
            case "updating" :
                self._add_updating(value)
            case "picking" : 
                self._add_picking(value)
            case "fps" :
                self._add_fps(value)

    def _add_rendering(self, value : float):
        self.rendering += value
        self.n_rendering += 1
    
    def _add_updating(self, value : float):
        self.updating += value
        self.n_updating += 1

    def _add_picking(self, value : float):
        self.picking += value
        self.n_picking += 1

    def _add_fps(self, value : float):
        self.fps += value
        self.n_fps += 1

    def get_values(self) -> dict :
        return {"rendering" : self.rendering / self.n_rendering if self.n_rendering != 0 else 0,
                "updating" : self.updating / self.n_updating if self.n_updating != 0 else 0,
                "picking" : self.picking / self.n_picking if self.n_picking != 0 else 0,
                "fps" : self.fps / self.n_fps if self.n_fps != 0 else 0}

        
def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

def read_file(path : str) -> list[tuple[Configuration, SampleMean]]:
    file = open(path, 'r')
    values : list[tuple[Configuration, SampleMean]] = []
    current_config = Configuration("default", 0, 0)
    current_sample = SampleMean()
    for line in file:
        line = line.split(" ")[0]
        elements = line.split("/")
        match elements[0]:
            case "PERF" :
                current_sample.add_value(elements[1], float(elements[2]))
            case "MODEL":
                if (current_config.is_valid()):
                    values.append((current_config, current_sample))
                current_sample = SampleMean()
                current_config = Configuration(elements[1], current_config.nb_element, current_config.nb_channel)
            case "SHAPE":
                if (current_config.is_valid()):
                    values.append((current_config, current_sample))
                current_sample = SampleMean()
                current_config = Configuration(current_config.model, elements[1], elements[2])
    values.append((current_config, current_sample))

    x = []
    y = []
    for config, sample in values:
        if (config.model == "Gol"):
            x.append(float(config.nb_element))
            y.append(sample.get_values()["rendering"])
        print("Config : ", config.get_config())
        print("Sample : ", sample.get_values())



    plt.xlabel("Number of elements (log)")
    plt.ylabel("Time (ms)")
    plt.xscale("log")
    plt.plot(x, y, color='green', marker='o', linestyle='dashed', label="test")
    print(x, y)
    plt.show()  
    plt.savefig("bro.png")


read_file(FILE)