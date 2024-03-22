from selenium import webdriver
from selenium.webdriver import Firefox

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities   
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import json


NAME = sys.argv[1]

binary_location = "/snap/bin/geckodriver"





service = webdriver.FirefoxService(executable_path=binary_location)

driver = Firefox(service=service)
driver.get("http://127.0.0.1:8000/")


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



def test_nb_elements(sizes : list[int], driver : Firefox) -> list[tuple[Configuration, SampleMean]]:
    button_reset = get_reset_button(driver)
    button_play = get_play_button(driver)
    grid_size = get_grid_size(driver)
    ret : list[tuple[Configuration, SampleMean]] = []
    for i in sizes:
        grid_size.clear()
        grid_size.send_keys(str(i))
        button_reset.click()
        button_play.click()
        print(grid_size.get_attribute("value"))
        NB_UPDATE = 5

        sample = SampleMean()
        config = Configuration(get_model_selector(driver).all_selected_options[0].text, i * i, 1)
        i = 0
        current_update = get_updating_time(driver)
        while(i < NB_UPDATE):
            update = get_updating_time(driver)
            if (update == current_update):
                time.sleep(0.1)
                continue
            current_update = update
            sample.add_value("updating", float(current_update))
            sample.add_value("rendering", float(get_rendering_time(driver)))
            i += 1
        ret.append((config, sample))
    return ret







def test_models(models : list[str], driver : Firefox, sleep : int = 3):
    time.sleep(3)
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.F12)
    ActionChains(driver).key_down(Keys.F12).key_up(Keys.F12).perform()
    selector = get_model_selector(driver)
    get_button_speed(driver).click()
    timer = get_input_timer(driver)
    timer.clear()

    ret : list[tuple[Configuration, SampleMean]] = []
    for i in range(2):
        timer.send_keys(Keys.LEFT)
    for model in models:
        selector.select_by_value(model)
        time.sleep(1)

        random = get_random_button(driver)
        if (not random.is_selected()):
            random.click()

        print(model)
        ret += test_nb_elements([10, 100, 200, 300, 400, 500, 600, 700, 800], driver)
    return ret

def get_reset_button(driver : Firefox):
    return driver.find_element(By.ID, "buttonReset")

def get_play_button(driver : Firefox):
    return driver.find_element(By.ID, "buttonPlay")

def get_pause_button(driver : Firefox):
    return driver.find_element(By.ID, "buttonPause")

def get_button_speed(driver : Firefox):
    return driver.find_element(By.ID, "buttonSpeed")

def get_grid_size(driver : Firefox):
    return driver.find_element(By.CSS_SELECTOR, "input[paramid=gridSize]")

def get_random_button(driver : Firefox):
    return driver.find_element(By.CSS_SELECTOR, "input[paramid=randomStart]")

def get_model_selector(driver : Firefox):
    return Select(driver.find_element(By.ID, "modelSelector"))

def get_input_timer(driver : Firefox):
    return driver.find_element(By.ID, "inputTimer")

def get_updating_time(driver : Firefox):
    el = driver.find_element(By.ID, "updateMs")
    return el.text.split(": ")[1].split(" ")[0]

def get_rendering_time(driver : Firefox):
    el = driver.find_element(By.ID, "renderingMs")
    return el.text.split(": ")[1].split(" ")[0]

def get_total_total(driver : Firefox):
    el = driver.find_element(By.ID, "totalMs")
    return el.text.split(": ")[1].split(" ")[0]

values = test_models(["Gol", "Lenia"], driver, 5)


def print_perf(values, models : list[str], name : str):
    plt.xlabel("Number of elements")
    plt.ylabel("Time (ms)")
    # plt.xscale("log")
    for model in models:
        x = []
        y = []
        for config, sample in values:
            if (config.model == model):
                x.append(config.nb_element)
                y.append(sample.get_values()["rendering"])

        plt.plot(x, y, marker='o', linestyle='dashed', label=model)
    plt.legend()
    plt.savefig("./" + "rendering_" + name + ".png")
    plt.cla()

    plt.xlabel("Number of elements")
    plt.ylabel("Time (ms)")
    # plt.xscale("log")
    for model in models:
        x = []
        y = []
        for config, sample in values:
            if (config.model == model):
                x.append(config.nb_element)
                y.append(sample.get_values()["updating"])

        plt.plot(x, y, marker='o', linestyle='dashed', label=model)
    plt.xlim(min(x), max(x))
    plt.legend()
    plt.savefig("./" + "updating_" + name + ".png")
    plt.cla()




    configs = []
    samples = []
    for config, sample in values:
        configs.append(config.get_config())
        samples.append(sample.get_values())

    with open("./" + name + ".json", "w") as outfile: 
        json.dump(configs, outfile)
        json.dump(samples, outfile)


print_perf(values, ["Gol", "Lenia"], NAME)

driver.quit()

print("DONE !")

