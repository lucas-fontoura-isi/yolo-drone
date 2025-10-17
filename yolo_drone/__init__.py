from . import __meta__
from .src import clean_data, slicing, segmentation

__version__ = __meta__.version
__all__ = ['clean_data', 'slicing', 'segmentation']