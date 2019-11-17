import numpy as np
from matplotlib import pyplot as plt
from .env_loader import EnvFactory
from .isovelocity import IsoBuilder
from .swellex import SwellexBuilder
from .deepwater import DeepWaterBuilder
'''
Description:
Initialize the factory and register all the builders

Author: Hunter Akins
'''

factory = EnvFactory()
factory.register_builder('iso', IsoBuilder)
factory.register_builder('swellex', SwellexBuilder)
factory.register_builder('deepwater', DeepWaterBuilder)

