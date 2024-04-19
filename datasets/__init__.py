from .sonn import ScanObjectNN, ScanObjectNN_hardest
from .mn import ModelNet, ModelNetFewShot

_datasets = {
    'sonn': ScanObjectNN,
    'sonn_hard': ScanObjectNN_hardest,
    'mn40': ModelNet,
    'mnfs': ModelNetFewShot,
}