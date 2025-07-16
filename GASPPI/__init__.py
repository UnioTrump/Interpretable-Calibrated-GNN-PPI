print("Loading GASPPI package...")
from .model.base import ScalableGNN
from .model.PPI import PPI

print("GASPPI package loaded successfully!")

__all__ = [
    'metis',
    'ScalableGNN',
    'PPI'
]