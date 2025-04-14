import importlib
import inspect
from utils import map_module2lib, inspect_api_info, validate_api_existence
api_path = 'torch.nn.functional.softmax'
info = inspect_api_info('torch.nn.functional', 'softmax')
for key, value in info.items():
    print(f'{key}: {value}')
