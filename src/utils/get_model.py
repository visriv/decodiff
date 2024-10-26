from src.model.diffusion import *
from src.model.diffusion_denoise import *
from src.model.pde_refiner import *
from src.model.diffusion_direct import *

def get_model(config):
    if (config.model.family_name == 'DiffusionModel'):
        model = DiffusionModel(config)
    elif (config.model.family_name == 'DiffusionDenoise'):
        model = DiffusionDenoise(config)
    elif (config.model.family_name == 'DiffusionDirect'):
        model = DiffusionDirect(config)
    elif (config.model.family_name == 'PdeRefiner'):
        model = PDERefiner(config)
    

    return model