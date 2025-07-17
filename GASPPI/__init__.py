print("Loading GASPPI package...")
from .model.base import ScalableGNN

print("GASPPI package loaded successfully!")

__all__ = [
    'metis',
    'ScalableGNN'
]