import time 


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Timer():
    
    def __init__(self, name : str):
        self.name : str = name
        self.t0 : float = None
        self.t1 : float = None

    def start(self):
        self.t0 = time.time()

    def stop(self, display : bool = True):
        self.t1 = time.time()
        if (display):
            print(self.name, " : ", bcolors.WARNING, 1000 * (self.t1 - self.t0), bcolors.ENDC, " ms")
    