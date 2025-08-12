from .base import GNNEncoder, ProteinGNN
from .dual_stream import DualStreamPPI
from .feature_stream_only import FeatureStreamOnlyPPI

__all__ = [
    'GNNEncoder',
    'ProteinGNN',
    'DualStreamPPI',
    'FeatureStreamOnlyPPI',
]