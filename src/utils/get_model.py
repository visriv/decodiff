from src.model.diffusion import *
from src.model.pde_refiner import *

def get_model(config):
    if (config.model.family_name == 'DiffusionModel'):
        model = DiffusionModel(config)
    elif (config.model.family_name == 'PdeRefiner'):
        model = PDERefiner(config)
    

    return model