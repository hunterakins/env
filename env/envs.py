import numpy as np
from matplotlib import pyplot as plt
from env.env.env_loader import EnvFactory
from env.env.isovelocity import IsoBuilder
from env.env.swellex_env import SwellexBuilder
from env.env.deepwater import DeepWaterBuilder
from env.env.swmfex_env import SWMFEXBuilder
'''
Description:
Initialize the factory and register all the builders

Author: Hunter Akins
'''

factory = EnvFactory()
factory.register_builder('iso', IsoBuilder)
factory.register_builder('swellex', SwellexBuilder)
factory.register_builder('deepwater', DeepWaterBuilder)
factory.register_builder('swmfex', SWMFEXBuilder)

