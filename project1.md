# AI Learns to Play Super Mario Bros!
## [程式碼來源](https://github.com/Chrispresso/SuperMarioBros-AI)
## [圖片來源為Youtube截圖](https://www.youtube.com/watch?v=CI3FRsSAa_U&t=125s&ab_channel=Chrispresso)

# Super Mario Bros
## 遊戲玩法
1. 玩家操控馬力歐後退、前進、跳躍
2. 躲避怪物、懸崖
3. 到達終點

![](https://github.com/yucing/ai111b/blob/main/picture/1.gif)

# 解析
* 將畫面轉換成 16*16 pixel section的圖

![](https://github.com/yucing/ai111b/blob/main/picture/1.png)

* 轉換完的圖為 15 * 16 (實際上為 13 * 16, 2行為顯示分數及時間)

![](https://github.com/yucing/ai111b/blob/main/picture/2.png)

![](https://github.com/yucing/ai111b/blob/main/picture/6.png)

* 下圖粉紅色邊框框起來的位置，為AI判斷輸入的位置

![](https://github.com/yucing/ai111b/blob/main/picture/3.png)

* 最後得出來的公式

![](https://github.com/yucing/ai111b/blob/main/picture/4.png)

# 遇到的問題
1. 剛開始，Mario會不知要怎麼動，因此採用 random 去操作
2. 當粉紅色區域沒有變更時，可能會被困住

![](https://github.com/yucing/ai111b/blob/main/picture/5.png)

3. 訓練模型不夠，無法做出跳躍、躲避怪物等動作

# 訓練結果
* 通關一次後，下一關跳過障礙物的延遲變小
## During the three weeks of trainning:
1. A total of 12048830 Marios were killed
2. These AI had a total playtime of just over 5 years
3. A combined 1422209 Marios beat a level

# Result
## [GIF來源](https://github.com/Chrispresso/SuperMarioBros-AI)
## 1-1

![](https://github.com/yucing/ai111b/blob/main/picture/1_1.gif)

## 4-1

![](https://github.com/yucing/ai111b/blob/main/picture/4_1.gif)

# 程式碼
## 馬力歐
```py
import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import random
import os
import csv

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
from config import Config



class Mario(Individual):
    def __init__(self,
                 config: Config,
                 chromosome: Optional[Dict[str, np.ndarray]] = None,
                 hidden_layer_architecture: List[int] = [12, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Union[int, float] = np.inf,
                 name: Optional[str] = None,
                 debug: Optional[bool] = False,
                 ):
        
        self.config = config

        self.lifespan = lifespan
        self.name = name
        self.debug = debug

        self._fitness = 0  # Overall fitness
        self._frames_since_progress = 0  # Number of frames since Mario has made progress towards the goal
        self._frames = 0  # Number of frames Mario has been alive
        
        self.hidden_layer_architecture = self.config.NeuralNetwork.hidden_layer_architecture
        self.hidden_activation = self.config.NeuralNetwork.hidden_node_activation
        self.output_activation = self.config.NeuralNetwork.output_node_activation

        self.start_row, self.viz_width, self.viz_height = self.config.NeuralNetwork.input_dims

        if self.config.NeuralNetwork.encode_row:
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height
        # print(f'num inputs:{num_inputs}')
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # Input Nodes
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden Layer Ndoes
        self.network_architecture.append(6)                        # 6 Outputs ['u', 'd', 'l', 'r', 'a', 'b']

        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
                                         )

        # If chromosome is set, take it
        if chromosome:
            self.network.params = chromosome
        
        self.is_alive = True
        self.x_dist = None
        self.game_score = None
        self.did_win = False
        # This is mainly just to "see" Mario winning
        self.allow_additional_time  = self.config.Misc.allow_additional_time_for_flagpole
        self.additional_timesteps = 0
        self.max_additional_timesteps = int(60*2.5)
        self._printed = False

        # Keys correspond with             B, NULL, SELECT, START, U, D, L, R, A
        # index                            0  1     2       3      4  5  6  7  8
        self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.farthest_x = 0


    @property
    def fitness(self):
        return self._fitness

    @property
    def chromosome(self):
        pass

    def decode_chromosome(self):
        pass

    def encode_chromosome(self):
        pass

    def calculate_fitness(self):
        frames = self._frames
        distance = self.x_dist
        score = self.game_score

        self._fitness = self.config.GeneticAlgorithm.fitness_func(frames, distance, score, self.did_win)

    def set_input_as_array(self, ram, tiles) -> None:
        mario_row, mario_col = SMB.get_mario_row_col(ram)
        arr = []
        
        for row in range(self.start_row, self.start_row + self.viz_height):
            for col in range(mario_col, mario_col + self.viz_width):
                try:
                    t = tiles[(row, col)]
                    if isinstance(t, StaticTileType):
                        if t.value == 0:
                            arr.append(0)
                        else:
                            arr.append(1)
                    elif isinstance(t, EnemyType):
                        arr.append(-1)
                    else:
                        raise Exception("This should never happen")
                except:
                    t = StaticTileType(0x00)
                    arr.append(0) # Empty

        self.inputs_as_array[:self.viz_height*self.viz_width, :] = np.array(arr).reshape((-1,1))
        if self.config.NeuralNetwork.encode_row:
            # Assign one-hot for mario row
            row = mario_row - self.start_row
            one_hot = np.zeros((self.viz_height, 1))
            if row >= 0 and row < self.viz_height:
                one_hot[row, 0] = 1
            self.inputs_as_array[self.viz_height*self.viz_width:, :] = one_hot.reshape((-1, 1))

    def update(self, ram, tiles, buttons, ouput_to_buttons_map) -> bool:
        """
        The main update call for Mario.
        Takes in inputs of surrounding area and feeds through the Neural Network
        
        Return: True if Mario is alive
                False otherwise
        """
        if self.is_alive:
            self._frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            # Sliding down flag pole
            if ram[0x001D] == 3:
                self.did_win = True
                if not self._printed and self.debug:
                    name = 'Mario '
                    name += f'{self.name}' if self.name else ''
                    print(f'{name} won')
                    self._printed = True
                if not self.allow_additional_time:
                    self.is_alive = False
                    return False
            # If we made it further, reset stats
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += 1

            if self.allow_additional_time and self.did_win:
                self.additional_timesteps += 1
            
            if self.allow_additional_time and self.additional_timesteps > self.max_additional_timesteps:
                self.is_alive = False
                return False
            elif not self.did_win and self._frames_since_progress > 60*3:
                self.is_alive = False
                return False            
        else:
            return False

        # Did you fly into a hole?
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        self.set_input_as_array(ram, tiles)

        # Calculate the output
        output = self.network.feed_forward(self.inputs_as_array)
        threshold = np.where(output > 0.5)[0]
        self.buttons_to_press.fill(0)  # Clear

        # Set buttons
        for b in threshold:
            self.buttons_to_press[ouput_to_buttons_map[b]] = 1

        return True
    
def save_mario(population_folder: str, individual_name: str, mario: Mario) -> None:
    # Make population folder if it doesnt exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # Save settings.config
    if 'settings.config' not in os.listdir(population_folder):
        with open(os.path.join(population_folder, 'settings.config'), 'w') as config_file:
            config_file.write(mario.config._config_text_file)
    
    # Make a directory for the individual
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    L = len(mario.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        b_name = 'b' + str(l)

        weights = mario.network.params[w_name]
        bias = mario.network.params[b_name]

        np.save(os.path.join(individual_dir, w_name), weights)
        np.save(os.path.join(individual_dir, b_name), bias)
    
def load_mario(population_folder: str, individual_name: str, config: Optional[Config] = None) -> Mario:
    # Make sure individual exists inside population folder
    if not os.path.exists(os.path.join(population_folder, individual_name)):
        raise Exception(f'{individual_name} not found inside {population_folder}')

    # Load a config if one is not given
    if not config:
        settings_path = os.path.join(population_folder, 'settings.config')
        config = None
        try:
            config = Config(settings_path)
        except:
            raise Exception(f'settings.config not found under {population_folder}')

    chromosome: Dict[str, np.ndarray] = {}
    # Grab all .npy files, i.e. W1.npy, b1.npy, etc. and load them into the chromosome
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            chromosome[param] = np.load(os.path.join(population_folder, individual_name, fname))
        
    mario = Mario(config, chromosome=chromosome)
    return mario

def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))

    return (mean, median, std, _min, _max)

def save_stats(population: Population, fname: str):
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = fname

    frames = [individual._frames for individual in population.individuals]
    max_distance = [individual.farthest_x for individual in population.individuals]
    fitness = [individual.fitness for individual in population.individuals]
    wins = [sum([individual.did_win for individual in population.individuals])]

    write_header = True
    if os.path.exists(f):
        write_header = False

    trackers = [('frames', frames),
                ('distance', max_distance),
                ('fitness', fitness),
                ('wins', wins)
                ]

    stats = ['mean', 'median', 'std', 'min', 'max']

    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(f, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',')
        if write_header:
            writer.writeheader()

        row = {}
        # Create a row to insert into csv
        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat

        # Write row
        writer.writerow(row)

def load_stats(path_to_stats: str, normalize: Optional[bool] = False):
    data = {}

    fieldnames = None
    trackers_stats = None
    trackers = None
    stats_names = None

    with open(path_to_stats, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = reader.fieldnames
        trackers_stats = [f.split('_') for f in fieldnames]
        trackers = set(ts[0] for ts in trackers_stats)
        stats_names = set(ts[1] for ts in trackers_stats)
        
        for tracker, stat_name in trackers_stats:
            if tracker not in data:
                data[tracker] = {}
            
            if stat_name not in data[tracker]:
                data[tracker][stat_name] = []

        for line in reader:
            for tracker in trackers:
                for stat_name in stats_names:
                    value = float(line['{}_{}'.format(tracker, stat_name)])
                    data[tracker][stat_name].append(value)
        
    if normalize:
        factors = {}
        for tracker in trackers:
            factors[tracker] = {}
            for stat_name in stats_names:
                factors[tracker][stat_name] = 1.0

        for tracker in trackers:
            for stat_name in stats_names:
                max_val = max([abs(d) for d in data[tracker][stat_name]])
                if max_val == 0:
                    max_val = 1
                factors[tracker][stat_name] = float(max_val)

        for tracker in trackers:
            for stat_name in stats_names:
                factor = factors[tracker][stat_name]
                d = data[tracker][stat_name]
                data[tracker][stat_name] = [val / factor for val in d]

    return data

def get_num_inputs(config: Config) -> int:
    _, viz_width, viz_height = config.NeuralNetwork.input_dims
    if config.NeuralNetwork.encode_row:
        num_inputs = viz_width * viz_height + viz_height
    else:
        num_inputs = viz_width * viz_height
    return num_inputs

def get_num_trainable_parameters(config: Config) -> int:
    num_inputs = get_num_inputs(config)
    hidden_layers = config.NeuralNetwork.hidden_layer_architecture
    num_outputs = 6  # U, D, L, R, A, B

    layers = [num_inputs] + hidden_layers + [num_outputs]
    num_params = 0
    for i in range(0, len(layers)-1):
        L      = layers[i]
        L_next = layers[i+1]
        num_params += L*L_next + L_next

    return num_params
```

# 神經網路
```py
import numpy as np
from typing import List, Callable, NewType, Optional


ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])

sigmoid = ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))
tanh = ActivationFunction(lambda X: np.tanh(X))
relu = ActivationFunction(lambda X: np.maximum(0, X))
leaky_relu = ActivationFunction(lambda X: np.where(X > 0, X, X * 0.01))
linear = ActivationFunction(lambda X: X)



class FeedForwardNetwork(object):
    def __init__(self,
                 layer_nodes: List[int],
                 hidden_activation: ActivationFunction,
                 output_activation: ActivationFunction,
                 init_method: Optional[str] = 'uniform',
                 seed: Optional[int] = None):
        self.params = {}
        self.layer_nodes = layer_nodes
        # print(self.layer_nodes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.inputs = None
        self.out = None

        self.rand = np.random.RandomState(seed)

        # Initialize weights and bias
        for l in range(1, len(self.layer_nodes)):
            if init_method == 'uniform':
                self.params['W' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], self.layer_nodes[l-1]))
                self.params['b' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], 1))
            
            else:
                raise Exception('Implement more options, bro')

            self.params['A' + str(l)] = None
        
        
    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        A_prev = X
        L = len(self.layer_nodes) - 1  # len(self.params) // 2

        # Feed hidden layers
        for l in range(1, L):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A_prev = self.hidden_activation(Z)
            self.params['A' + str(l)] = A_prev

        # Feed output
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A_prev) + b
        out = self.output_activation(Z)
        self.params['A' + str(L)] = out

        self.out = out
        return out

    def softmax(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) / np.sum(np.exp(X), axis=0)

def get_activation_by_name(name: str) -> ActivationFunction:
    activations = [('relu', relu),
                   ('sigmoid', sigmoid),
                   ('linear', linear),
                   ('leaky_relu', leaky_relu),
                   ('tanh', tanh),
    ]

    func = [activation[1] for activation in activations if activation[0].lower() == name.lower()]
    assert len(func) == 1

    return func[0]
```

# 配置
```py
import configparser
import os
from typing import Any, Dict


# A mapping from parameters name -> final type
_params = {
    # Graphics Params
    'Graphics': {
        'tile_size': (tuple, float),
        'neuron_radius': float,
    },

    # Statistics Params
    'Statistics': {
        'save_best_individual_from_generation': str,
        'save_population_stats': str,
    },

    # NeuralNetwork Params
    'NeuralNetwork': {
        'input_dims': (tuple, int),
        'hidden_layer_architecture': (tuple, int),
        'hidden_node_activation': str,
        'output_node_activation': str,
        'encode_row': bool,
    },

    # Genetic Algorithm
    'GeneticAlgorithm': {
        'fitness_func': type(lambda : None)
    },

    # Crossover Params
    'Crossover': {
        'probability_sbx': float,
        'sbx_eta': float,
        'crossover_selection': str,
        'tournament_size': int,
    },

    # Mutation Params
    'Mutation': {
        'mutation_rate': float,
        'mutation_rate_type': str,
        'gaussian_mutation_scale': float,
    },

    # Selection Params
    'Selection': {
        'num_parents': int,
        'num_offspring': int,
        'selection_type': str,
        'lifespan': float
    },

    # Misc Params
    'Misc': {
        'level': str,
        'allow_additional_time_for_flagpole': bool
    }
}

class DotNotation(object):
    def __init__(self, d: Dict[Any, Any]):
        for k in d:
            # If the key is another dictionary, keep going
            if isinstance(d[k], dict):
                self.__dict__[k] = DotNotation(d[k])
            # If it's a list or tuple then check to see if any element is a dictionary
            elif isinstance(d[k], (list, tuple)):
                l = []
                for v in d[k]:
                    if isinstance(v, dict):
                        l.append(DotNotation(v))
                    else:
                        l.append(v)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = d[k]
    
    def __getitem__(self, name) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)


class Config(object):
    def __init__(self,
                 filename: str
                 ):
        self.filename = filename
        
        if not os.path.isfile(self.filename):
            raise Exception('No file found named "{}"'.format(self.filename))

        with open(self.filename) as f:
            self._config_text_file = f.read()

        self._config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._config.read(self.filename)

        self._verify_sections()
        self._create_dict_from_config()
        self._set_dict_types()
        dot_notation = DotNotation(self._config_dict)
        self.__dict__.update(dot_notation.__dict__)


    def _create_dict_from_config(self) -> None:
        d = {}
        for section in self._config.sections():
            d[section] = {}
            for k, v in self._config[section].items():
                d[section][k] = v

        self._config_dict = d

    def _set_dict_types(self) -> None:
        for section in self._config_dict:
            for k, v in self._config_dict[section].items():
                try:
                    _type = _params[section][k]
                except:
                    raise Exception('No value "{}" found for section "{}". Please set this in _params'.format(k, section))
                # Normally _type will be int, str, float or some type of built-in type.
                # If _type is an instance of a tuple, then we need to split the data
                if isinstance(_type, tuple):
                    if len(_type) == 2:
                        cast = _type[1]
                        v = v.replace('(', '').replace(')', '')  # Remove any parens that might be present 
                        self._config_dict[section][k] = tuple(cast(val) for val in v.split(','))
                    else:
                        raise Exception('Expected a 2 tuple value describing that it is to be parse as a tuple and the type to cast it as')
                elif 'lambda' in v:
                    try:
                        self._config_dict[section][k] = eval(v)
                    except:
                        pass
                # Is it a bool?
                elif _type == bool:
                    self._config_dict[section][k] = _type(eval(v))
                # Otherwise parse normally
                else:
                    self._config_dict[section][k] = _type(v)

    def _verify_sections(self) -> None:
        # Validate sections
        for section in self._config.sections():
            # Make sure the section is allowed
            if section not in _params:
                raise Exception('Section "{}" has no parameters allowed. Please remove this section and run again.'.format(section))

    def _get_reference_from_dict(self, reference: str) -> Any:
        path = reference.split('.')
        d = self._config_dict
        for p in path:
            d = d[p]
        
        assert type(d) in (tuple, int, float, bool, str)
        return d

    def _is_number(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False
```

# smb
```py
import retro
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPolygonF, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF, QTimer, QRect
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from PIL import Image
from PIL.ImageQt import ImageQt
from typing import Tuple, List, Optional
import random
import sys
import math
import numpy as np
import argparse
import os

from utils import SMB, EnemyType, StaticTileType, ColorMap, DynamicTileType
from config import Config
from nn_viz import NeuralNetworkViz
from mario import Mario, save_mario, save_stats, get_num_trainable_parameters, get_num_inputs, load_mario

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, tournament_selection, roulette_wheel_selection
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation

normal_font = QtGui.QFont('Times', 11, QtGui.QFont.Normal)
font_bold = QtGui.QFont('Times', 11, QtGui.QFont.Bold)


def draw_border(painter: QPainter, size: Tuple[float, float]) -> None:
    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
    painter.setBrush(QBrush(Qt.green, Qt.NoBrush))
    painter.setRenderHint(QPainter.Antialiasing)
    points = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    qpoints = [QPointF(point[0], point[1]) for point in points]
    polygon = QPolygonF(qpoints)
    painter.drawPolygon(polygon)


class Visualizer(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config, nn_viz: NeuralNetworkViz):
        super().__init__(parent)
        self.size = size
        self.config = config
        self.nn_viz = nn_viz
        self.ram = None
        self.x_offset = 150
        self.tile_width, self.tile_height = self.config.Graphics.tile_size
        self.tiles = None
        self.enemies = None
        self._should_update = True

    def _draw_region_of_interest(self, painter: QPainter) -> None:
        # Grab mario row/col in our tiles
        mario = SMB.get_mario_location_on_screen(self.ram)
        mario_row, mario_col = SMB.get_mario_row_col(self.ram)
        x = mario_col
       
        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))

        start_row, viz_width, viz_height = self.config.NeuralNetwork.input_dims
        painter.drawRect(x*self.tile_width + 5 + self.x_offset, start_row*self.tile_height + 5, viz_width*self.tile_width, viz_height*self.tile_height)


    def draw_tiles(self, painter: QPainter):
        if not self.tiles:
            return
        for row in range(15):
            for col in range(16):
                painter.setPen(QPen(Qt.black,  1, Qt.SolidLine))
                painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                x_start = 5 + (self.tile_width * col) + self.x_offset
                y_start = 5 + (self.tile_height * row)

                loc = (row, col)
                tile = self.tiles[loc]

                if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                    rgb = ColorMap[tile.name].value
                    color = QColor(*rgb)
                    painter.setBrush(QBrush(color))
                else:
                    pass

                painter.drawRect(x_start, y_start, self.tile_width, self.tile_height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)

        if self._should_update:
            draw_border(painter, self.size)
            if not self.ram is None:
                self.draw_tiles(painter)
                self._draw_region_of_interest(painter)
                self.nn_viz.show_network(painter)
        else:
            # draw_border(painter, self.size)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QtGui.QFont('Times', 30, QtGui.QFont.Normal))
            txt = 'Display is hidden.\nHit Ctrl+V to show\nConfig: {}'.format(args.config)
            painter.drawText(event.rect(), Qt.AlignCenter, txt)
            pass

        painter.end()

    def _update(self):
        self.update()


class GameWindow(QtWidgets.QWidget):
    def __init__(self, parent, size, config: Config):
        super().__init__(parent)
        self._should_update = True
        self.size = size
        self.config = config
        self.screen = None
        self.img_label = QtWidgets.QLabel(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.img_label)
        self.setLayout(self.layout)
        

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        if self._should_update:
            draw_border(painter, self.size)
            if not self.screen is None:
                # self.img_label = QtWidgets.QLabel(self.centralWidget)
                # screen = self.env.reset()
    
                width = self.screen.shape[0] * 3 
                height = int(self.screen.shape[1] * 2)
                resized = self.screen
                original = QImage(self.screen, self.screen.shape[1], self.screen.shape[0], QImage.Format_RGB888)
                # Create the image and label
                qimage = QImage(original)
                # Center where the image will go
                x = (self.screen.shape[0] - width) // 2
                y = (self.screen.shape[1] - height) // 2
                self.img_label.setGeometry(0, 0, width, height)
                # Add image
                pixmap = QPixmap(qimage)
                pixmap = pixmap.scaled(width, height, Qt.KeepAspectRatio)
                self.img_label.setPixmap(pixmap)
        else:
            self.img_label.clear()
            # draw_border(painter, self.size)
        painter.end()

    def _update(self):
        self.update()

class InformationWidget(QtWidgets.QWidget):
    def __init__(self, parent, size, config):
        super().__init__(parent)
        self.size = size
        self.config = config

        self.grid = QtWidgets.QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self._init_window()
        # self.grid.setSpacing(20)
        self.setLayout(self.grid)


    def _init_window(self) -> None:
        info_vbox = QVBoxLayout()
        info_vbox.setContentsMargins(0, 0, 0, 0)
        ga_vbox = QVBoxLayout()
        ga_vbox.setContentsMargins(0, 0, 0, 0)

        # Current Generation
        generation_label = QLabel()
        generation_label.setFont(font_bold)
        generation_label.setText('Generation:')
        generation_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.generation = QLabel()
        self.generation.setFont(normal_font)
        self.generation.setText("<font color='red'>" + '1' + '</font>')
        self.generation.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_generation = QHBoxLayout()
        hbox_generation.setContentsMargins(5, 0, 0, 0)
        hbox_generation.addWidget(generation_label, 1)
        hbox_generation.addWidget(self.generation, 1)
        info_vbox.addLayout(hbox_generation)

        # Current individual
        current_individual_label = QLabel()
        current_individual_label.setFont(font_bold)
        current_individual_label.setText('Individual:')
        current_individual_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.current_individual = QLabel()
        self.current_individual.setFont(normal_font)
        self.current_individual.setText('1/{}'.format(self.config.Selection.num_parents))
        self.current_individual.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_current_individual = QHBoxLayout()
        hbox_current_individual.setContentsMargins(5, 0, 0, 0)
        hbox_current_individual.addWidget(current_individual_label, 1)
        hbox_current_individual.addWidget(self.current_individual, 1)
        info_vbox.addLayout(hbox_current_individual)

        # Best fitness
        best_fitness_label = QLabel()
        best_fitness_label.setFont(font_bold)
        best_fitness_label.setText('Best Fitness:')
        best_fitness_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.best_fitness = QLabel()
        self.best_fitness.setFont(normal_font)
        self.best_fitness.setText('0')
        self.best_fitness.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_best_fitness = QHBoxLayout()
        hbox_best_fitness.setContentsMargins(5, 0, 0, 0)
        hbox_best_fitness.addWidget(best_fitness_label, 1)
        hbox_best_fitness.addWidget(self.best_fitness, 1)
        info_vbox.addLayout(hbox_best_fitness) 

        # Max Distance
        max_distance_label = QLabel()
        max_distance_label.setFont(font_bold)
        max_distance_label.setText('Max Distance:')
        max_distance_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.max_distance = QLabel()
        self.max_distance.setFont(normal_font)
        self.max_distance.setText('0')
        self.max_distance.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_max_distance = QHBoxLayout()
        hbox_max_distance.setContentsMargins(5, 0, 0, 0)
        hbox_max_distance.addWidget(max_distance_label, 1)
        hbox_max_distance.addWidget(self.max_distance, 1)
        info_vbox.addLayout(hbox_max_distance)

        # Num inputs
        num_inputs_label = QLabel()
        num_inputs_label.setFont(font_bold)
        num_inputs_label.setText('Num Inputs:')
        num_inputs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        num_inputs = QLabel()
        num_inputs.setFont(normal_font)
        num_inputs.setText(str(get_num_inputs(self.config)))
        num_inputs.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_num_inputs = QHBoxLayout()
        hbox_num_inputs.setContentsMargins(5, 0, 0, 0)
        hbox_num_inputs.addWidget(num_inputs_label, 1)
        hbox_num_inputs.addWidget(num_inputs, 1)
        info_vbox.addLayout(hbox_num_inputs)

        # Trainable params
        trainable_params_label = QLabel()
        trainable_params_label.setFont(font_bold)
        trainable_params_label.setText('Trainable Params:')
        trainable_params_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        trainable_params = QLabel()
        trainable_params.setFont(normal_font)
        trainable_params.setText(str(get_num_trainable_parameters(self.config)))
        trainable_params.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hbox_trainable_params = QHBoxLayout()
        hbox_trainable_params.setContentsMargins(5, 0, 0, 0)
        hbox_trainable_params.addWidget(trainable_params_label, 1)
        hbox_trainable_params.addWidget(trainable_params, 1)
        info_vbox.addLayout(hbox_trainable_params)

        # Selection
        selection_type = self.config.Selection.selection_type
        num_parents = self.config.Selection.num_parents
        num_offspring = self.config.Selection.num_offspring
        if selection_type == 'comma':
            selection_txt = '{}, {}'.format(num_parents, num_offspring)
        elif selection_type == 'plus':
            selection_txt = '{} + {}'.format(num_parents, num_offspring)
        else:
            raise Exception('Unkown Selection type "{}"'.format(selection_type))
        selection_hbox = self._create_hbox('Offspring:', font_bold, selection_txt, normal_font)
        ga_vbox.addLayout(selection_hbox)

        # Lifespan
        lifespan = self.config.Selection.lifespan
        lifespan_txt = 'Infinite' if lifespan == np.inf else str(lifespan)
        lifespan_hbox = self._create_hbox('Lifespan:', font_bold, lifespan_txt, normal_font)
        ga_vbox.addLayout(lifespan_hbox)

        # Mutation rate
        mutation_rate = self.config.Mutation.mutation_rate
        mutation_type = self.config.Mutation.mutation_rate_type.capitalize()
        mutation_txt = '{} {}% '.format(mutation_type, str(round(mutation_rate*100, 2)))
        mutation_hbox = self._create_hbox('Mutation:', font_bold, mutation_txt, normal_font)
        ga_vbox.addLayout(mutation_hbox)

        # Crossover
        crossover_selection = self.config.Crossover.crossover_selection
        if crossover_selection == 'roulette':
            crossover_txt = 'Roulette'
        elif crossover_selection == 'tournament':
            crossover_txt = 'Tournament({})'.format(self.config.Crossover.tournament_size)
        else:
            raise Exception('Unknown crossover selection "{}"'.format(crossover_selection))
        crossover_hbox = self._create_hbox('Crossover:', font_bold, crossover_txt, normal_font)
        ga_vbox.addLayout(crossover_hbox)

        # SBX eta
        sbx_eta_txt = str(self.config.Crossover.sbx_eta)
        sbx_hbox = self._create_hbox('SBX Eta:', font_bold, sbx_eta_txt, normal_font)
        ga_vbox.addLayout(sbx_hbox)

        # Layers
        num_inputs = get_num_inputs(self.config)
        hidden = self.config.NeuralNetwork.hidden_layer_architecture
        num_outputs = 6
        L = [num_inputs] + hidden + [num_outputs]
        layers_txt = '[' + ', '.join(str(nodes) for nodes in L) + ']'
        layers_hbox = self._create_hbox('Layers:', font_bold, layers_txt, normal_font)
        ga_vbox.addLayout(layers_hbox)

        self.grid.addLayout(info_vbox, 0, 0)
        self.grid.addLayout(ga_vbox, 0, 1)

    def _create_hbox(self, title: str, title_font: QtGui.QFont,
                     content: str, content_font: QtGui.QFont) -> QHBoxLayout:
        title_label = QLabel()
        title_label.setFont(title_font)
        title_label.setText(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        content_label = QLabel()
        content_label.setFont(content_font)
        content_label.setText(content)
        content_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(5, 0, 0, 0)
        hbox.addWidget(title_label, 1)
        hbox.addWidget(content_label, 1)
        return hbox


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        global args
        self.config = config
        self.top = 150
        self.left = 150
        self.width = 1100
        self.height = 700

        self.title = 'Super Mario Bros AI'
        self.current_generation = 0
        # This is the generation that is actual 0. If you load individuals then you might end up starting at gen 12, in which case
        # gen 12 would be the true 0
        self._true_zero_gen = 0

        self._should_display = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
        # index                0  1     2       3      4  5  6  7  8
        self.keys = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
        # We need a mapping from the output to the keys above
        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }

        # Initialize the starting population
        individuals: List[Individual] = []

        # Load any individuals listed in the args.load_inds
        num_loaded = 0
        if args.load_inds:
            # Overwrite the config file IF one is not specified
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.load_file, 'settings.config'))
                except:
                    raise Exception(f'settings.config not found under {args.load_file}')

            set_of_inds = set(args.load_inds)

            for ind_name in os.listdir(args.load_file):
                if ind_name.startswith('best_ind_gen'):
                    ind_number = int(ind_name[len('best_ind_gen'):])
                    if ind_number in set_of_inds:
                        individual = load_mario(args.load_file, ind_name, self.config)
                        # Set debug stuff if needed
                        if args.debug:
                            individual.name = f'm{num_loaded}_loaded'
                            individual.debug = True
                        individuals.append(individual)
                        num_loaded += 1
            
            # Set the generation
            self.current_generation = max(set_of_inds) + 1  # +1 becauase it's the next generation
            self._true_zero_gen = self.current_generation

        # Load any individuals listed in args.replay_inds
        if args.replay_inds:
            # Overwrite the config file IF one is not specified
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.replay_file, 'settings.config'))
                except:
                    raise Exception(f'settings.config not found under {args.replay_file}')

            for ind_gen in args.replay_inds:
                ind_name = f'best_ind_gen{ind_gen}'
                fname = os.path.join(args.replay_file, ind_name)
                if os.path.exists(fname):
                    individual = load_mario(args.replay_file, ind_name, self.config)
                    # Set debug stuff if needed
                    if args.debug:
                        individual.name= f'm_gen{ind_gen}_replay'
                        individual.debug = True
                    individuals.append(individual)
                else:
                    raise Exception(f'No individual named {ind_name} under {args.replay_file}')
        # If it's not a replay then we need to continue creating individuals
        else:
            num_parents = max(self.config.Selection.num_parents - num_loaded, 0)
            for _ in range(num_parents):
                individual = Mario(self.config)
                # Set debug stuff if needed
                if args.debug:
                    individual.name = f'm{num_loaded}'
                    individual.debug = True
                individuals.append(individual)
                num_loaded += 1

        self.best_fitness = 0.0
        self._current_individual = 0
        self.population = Population(individuals)

        self.mario = self.population.individuals[self._current_individual]
        
        self.max_distance = 0  # Track farthest traveled in level
        self.max_fitness = 0.0
        self.env = retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.Misc.level}')

        # Determine the size of the next generation based off selection type
        self._next_gen_size = None
        if self.config.Selection.selection_type == 'plus':
            self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
        elif self.config.Selection.selection_type == 'comma':
            self._next_gen_size = self.config.Selection.num_offspring

        # If we aren't displaying we need to reset the environment to begin with
        if args.no_display:
            self.env.reset()
        else:
            self.init_window()

            # Set the generation in the label if needed
            if args.load_inds:
                txt = "<font color='red'>" + str(self.current_generation + 1) + '</font>'  # +1 because we switch from 0 to 1 index
                self.info_window.generation.setText(txt)

            # if this is a replay then just set current_individual to be 'replay' and set generation
            if args.replay_file:
                self.info_window.current_individual.setText('Replay')
                txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
                self.info_window.generation.setText(txt)

            self.show()

        if args.no_display:
            self._timer.start(1000 // 1000)
        else:
            self._timer.start(1000 // 60)

    def init_window(self) -> None:
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        self.game_window = GameWindow(self.centralWidget, (514, 480), self.config)
        self.game_window.setGeometry(QRect(1100-514, 0, 514, 480))
        self.game_window.setObjectName('game_window')
        # # Reset environment and pass the screen to the GameWindow
        screen = self.env.reset()
        self.game_window.screen = screen
 
        self.viz = NeuralNetworkViz(self.centralWidget, self.mario, (1100-514, 700), self.config)

        self.viz_window = Visualizer(self.centralWidget, (1100-514, 700), self.config, self.viz)
        self.viz_window.setGeometry(0, 0, 1100-514, 700)
        self.viz_window.setObjectName('viz_window')
        self.viz_window.ram = self.env.get_ram()
        
        self.info_window = InformationWidget(self.centralWidget, (514, 700-480), self.config)
        self.info_window.setGeometry(QRect(1100-514, 480, 514, 700-480))

    def keyPressEvent(self, event):
        k = event.key()
        # m = {
        #     Qt.Key_Right : 7,
        #     Qt.Key_C : 8,
        #     Qt.Key_X: 0,
        #     Qt.Key_Left: 6,
        #     Qt.Key_Down: 5
        # }
        # if k in m:
        #     self.keys[m[k]] = 1
        # if k == Qt.Key_D:
        #     tiles = SMB.get_tiles(self.env.get_ram(), False)
        modifier = int(event.modifiers())
        if modifier == Qt.CTRL:
            if k == Qt.Key_V:
                self._should_display = not self._should_display

    def keyReleaseEvent(self, event):
        k = event.key()
        m = {
            Qt.Key_Right : 7,
            Qt.Key_C : 8,
            Qt.Key_X: 0,
            Qt.Key_Left: 6,
            Qt.Key_Down: 5
        }
        if k in m:
            self.keys[m[k]] = 0

        
    def next_generation(self) -> None:
        self._increment_generation()
        self._current_individual = 0

        if not args.no_display:
            self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, self._next_gen_size))

        # Calculate fitness
        # print(', '.join(['{:.2f}'.format(i.fitness) for i in self.population.individuals]))

        if args.debug:
            print(f'----Current Gen: {self.current_generation}, True Zero: {self._true_zero_gen}')
            fittest = self.population.fittest_individual
            print(f'Best fitness of gen: {fittest.fitness}, Max dist of gen: {fittest.farthest_x}')
            num_wins = sum(individual.did_win for individual in self.population.individuals)
            pop_size = len(self.population.individuals)
            print(f'Wins: {num_wins}/{pop_size} (~{(float(num_wins)/pop_size*100):.2f}%)')

        if self.config.Statistics.save_best_individual_from_generation:
            folder = self.config.Statistics.save_best_individual_from_generation
            best_ind_name = 'best_ind_gen{}'.format(self.current_generation - 1)
            best_ind = self.population.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        if self.config.Statistics.save_population_stats:
            fname = self.config.Statistics.save_population_stats
            save_stats(self.population, fname)

        self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)

        random.shuffle(self.population.individuals)
        next_pop = []

        # Parents + offspring
        if self.config.Selection.selection_type == 'plus':
            # Decrement lifespan
            for individual in self.population.individuals:
                individual.lifespan -= 1

            for individual in self.population.individuals:
                config = individual.config
                chromosome = individual.network.params
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan
                name = individual.name

                # If the indivdual would be alve, add it to the next pop
                if lifespan > 0:
                    m = Mario(config, chromosome, hidden_layer_architecture, hidden_activation, output_activation, lifespan)
                    # Set debug if needed
                    if args.debug:
                        m.name = f'{name}_life{lifespan}'
                        m.debug = True
                    next_pop.append(m)

        num_loaded = 0

        while len(next_pop) < self._next_gen_size:
            selection = self.config.Crossover.crossover_selection
            if selection == 'tournament':
                p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
            elif selection == 'roulette':
                p1, p2 = roulette_wheel_selection(self.population, 2)
            else:
                raise Exception('crossover_selection "{}" is not supported'.format(selection))

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]  
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                #  Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])


            c1 = Mario(self.config, c1_params, p1.hidden_layer_architecture, p1.hidden_activation, p1.output_activation, p1.lifespan)
            c2 = Mario(self.config, c2_params, p2.hidden_layer_architecture, p2.hidden_activation, p2.output_activation, p2.lifespan)

            # Set debug if needed
            if args.debug:
                c1_name = f'm{num_loaded}_new'
                c1.name = c1_name
                c1.debug = True
                num_loaded += 1

                c2_name = f'm{num_loaded}_new'
                c2.name = c2_name
                c2.debug = True
                num_loaded += 1

            next_pop.extend([c1, c2])

        # Set next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eta = self.config.Crossover.sbx_eta

        # SBX weights and bias
        child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
        child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, eta)

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        mutation_rate = self.config.Mutation.mutation_rate
        scale = self.config.Mutation.gaussian_mutation_scale

        if self.config.Mutation.mutation_rate_type == 'dynamic':
            mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)
        
        # Mutate weights
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

        # Mutate bias
        gaussian_mutation(child1_bias, mutation_rate, scale=scale)
        gaussian_mutation(child2_bias, mutation_rate, scale=scale)

    def _increment_generation(self) -> None:
        self.current_generation += 1
        if not args.no_display:
            txt = "<font color='red'>" + str(self.current_generation + 1) + '</font>'
            self.info_window.generation.setText(txt)


    def _update(self) -> None:
        """
        This is the main update method which is called based on the FPS timer.
        Genetic Algorithm updates, window updates, etc. are performed here.
        """
        ret = self.env.step(self.mario.buttons_to_press)

        if not args.no_display:
            if self._should_display:
                self.game_window.screen = ret[0]
                self.game_window._should_update = True
                self.info_window.show()
                self.viz_window.ram = self.env.get_ram()
            else:
                self.game_window._should_update = False
                self.info_window.hide()
            self.game_window._update()

        ram = self.env.get_ram()
        tiles = SMB.get_tiles(ram)  # Grab tiles on the screen
        enemies = SMB.get_enemy_locations(ram)

        # self.mario.set_input_as_array(ram, tiles)
        self.mario.update(ram, tiles, self.keys, self.ouput_to_keys_map)
        
        if not args.no_display:
            if self._should_display:
                self.viz_window.ram = ram
                self.viz_window.tiles = tiles
                self.viz_window.enemies = enemies
                self.viz_window._should_update = True
            else:
                self.viz_window._should_update = False
            self.viz_window._update()
    
        if self.mario.is_alive:
            # New farthest distance?
            if self.mario.farthest_x > self.max_distance:
                if args.debug:
                    print('New farthest distance:', self.mario.farthest_x)
                self.max_distance = self.mario.farthest_x
                if not args.no_display:
                    self.info_window.max_distance.setText(str(self.max_distance))
        else:
            self.mario.calculate_fitness()
            fitness = self.mario.fitness
            
            if fitness > self.max_fitness:
                self.max_fitness = fitness
                max_fitness = '{:.2f}'.format(self.max_fitness)
                if not args.no_display:
                    self.info_window.best_fitness.setText(max_fitness)
            # Next individual
            self._current_individual += 1

            # Are we replaying from a file?
            if args.replay_file:
                if not args.no_display:
                    # Set the generation to be whatever best individual is being ran (+1)
                    # Check to see if there is a next individual, otherwise exit
                    if self._current_individual >= len(args.replay_inds):
                        if args.debug:
                            print(f'Finished replaying {len(args.replay_inds)} best individuals')
                        sys.exit()

                    txt = f"<font color='red'>{args.replay_inds[self._current_individual] + 1}</font>"
                    self.info_window.generation.setText(txt)
            else:
                # Is it the next generation?
                if (self.current_generation > self._true_zero_gen and self._current_individual == self._next_gen_size) or\
                    (self.current_generation == self._true_zero_gen and self._current_individual == self.config.Selection.num_parents):
                    self.next_generation()
                else:
                    if self.current_generation == self._true_zero_gen:
                        current_pop = self.config.Selection.num_parents
                    else:
                        current_pop = self._next_gen_size
                    if not args.no_display:
                        self.info_window.current_individual.setText('{}/{}'.format(self._current_individual + 1, current_pop))
            
            if args.no_display:
                self.env.reset()
            else:
                self.game_window.screen = self.env.reset()
            
            self.mario = self.population.individuals[self._current_individual]

            if not args.no_display:
                self.viz.mario = self.mario
        

def parse_args():
    parser = argparse.ArgumentParser(description='Super Mario Bros AI')

    # Config
    parser.add_argument('-c', '--config', dest='config', required=False, help='config file to use')
    # Load arguments
    parser.add_argument('--load-file', dest='load_file', required=False, help='/path/to/population that you want to load individuals from')
    parser.add_argument('--load-inds', dest='load_inds', required=False, help='[start,stop] (inclusive) or ind1,ind2,... that you wish to load from the file')
    # No display
    parser.add_argument('--no-display', dest='no_display', required=False, default=False, action='store_true', help='If set, there will be no Qt graphics displayed and FPS is increased to max')
    # Debug
    parser.add_argument('--debug', dest='debug', required=False, default=False, action='store_true', help='If set, certain debug messages will be printed')
    # Replay arguments
    parser.add_argument('--replay-file', dest='replay_file', required=False, default=None, help='/path/to/population that you want to replay from')
    parser.add_argument('--replay-inds', dest='replay_inds', required=False, default=None, help='[start,stop] (inclusive) or ind1,ind2,ind50,... or [start,] that you wish to replay from file')

    args = parser.parse_args()
    
    load_from_file = bool(args.load_file) and bool(args.load_inds)
    replay_from_file = bool(args.replay_file) and bool(args.replay_inds)

    # Load from file checks
    if bool(args.load_file) ^ bool(args.load_inds):
        parser.error('--load-file and --load-inds must be used together.')
    if load_from_file:
        # Convert the load_inds to be a list
        # Is it a range?
        if '[' in args.load_inds and ']' in args.load_inds:
            args.load_inds = args.load_inds.replace('[', '').replace(']', '')
            ranges = args.load_inds.split(',')
            start_idx = int(ranges[0])
            end_idx = int(ranges[1])
            args.load_inds = list(range(start_idx, end_idx + 1))
        # Otherwise it's a list of individuals to load
        else:
            args.load_inds = [int(ind) for ind in args.load_inds.split(',')]

    # Replay from file checks
    if bool(args.replay_file) ^ bool(args.replay_inds):
        parser.error('--replay-file and --replay-inds must be used together.')
    if replay_from_file:
        # Convert the replay_inds to be a list
        # is it a range?
        if '[' in args.replay_inds and ']' in args.replay_inds:
            args.replay_inds = args.replay_inds.replace('[', '').replace(']', '')
            ranges = args.replay_inds.split(',')
            has_end_idx = bool(ranges[1])
            start_idx = int(ranges[0])
            # Is there an end idx? i.e. [12,15]
            if has_end_idx:
                end_idx = int(ranges[1])
                args.replay_inds = list(range(start_idx, end_idx + 1))
            # Or is it just a start? i.e. [12,]
            else:
                end_idx = start_idx
                for fname in os.listdir(args.replay_file):
                    if fname.startswith('best_ind_gen'):
                        ind_num = int(fname[len('best_ind_gen'):])
                        if ind_num > end_idx:
                            end_idx = ind_num
                args.replay_inds = list(range(start_idx, end_idx + 1))
        # Otherwise it's a list of individuals
        else:
            args.replay_inds = [int(ind) for ind in args.replay_inds.split(',')]

    if replay_from_file and load_from_file:
        parser.error('Cannot replay and load from a file.')

    # Make sure config AND/OR [(load_file and load_inds) or (replay_file and replay_inds)]
    if not (bool(args.config) or (load_from_file or replay_from_file)):
        parser.error('Must specify -c and/or [(--load-file and --load-inds) or (--replay-file and --replay-inds)]')

    return args

if __name__ == "__main__":
    global args
    args = parse_args()
    config = None
    if args.config:
        config = Config(args.config)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config)
    sys.exit(app.exec_())
```