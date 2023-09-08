import warnings

from torch import nn

from baal.bayesian.caching_utils import LRUCacheModule

WARNING_CACHE_REPLICATED = """
To use MCCachingModule at maximum effiency, we recommend using
 `replicate_in_memory=False`, but it is `True`.
"""


def raise_warnings_cache_replicated(module, replicate_in_memory):
    if (
        isinstance(module, nn.Module)
        and replicate_in_memory
        and any(isinstance(m, LRUCacheModule) for m in module.modules())
    ):
        warnings.warn(WARNING_CACHE_REPLICATED, UserWarning)
