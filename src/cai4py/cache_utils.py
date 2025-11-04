# python
from functools import lru_cache, wraps
from collections import OrderedDict
from typing import Callable, Any, Dict, Tuple
from collections import namedtuple

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


def flush_on_full_cache(maxsize: int):
    def decorator(fn: Callable):
        store: "OrderedDict[Tuple, Any]" = OrderedDict()
        hits = 0
        misses = 0

        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            key = (args, tuple(sorted(kwargs.items())))
            if key in store:
                store.move_to_end(key)  # treat as used
                hits += 1
                return store[key]
            misses += 1
            if len(store) >= maxsize:
                store.clear()  # flush entire cache on first insertion that would overflow
            value = fn(*args, **kwargs)
            store[key] = value
            return value

        def cache_info():
            return CacheInfo(hits, misses, maxsize, len(store))

        def cache_size():
            return len(store)

        # expose a tiny cache API similar to lru_cache
        wrapper.cache_clear = store.clear
        wrapper.cache_size = cache_size
        wrapper.cache_info = cache_info
        return wrapper

    return decorator


def make_cached_versions(fn: Callable, maxsize: int = 256):
    """Make a cached version of fn with both LRU and flush-on-full eviction policies."""
    # LRU version
    lru_wrapped = lru_cache(maxsize=maxsize)(fn)

    # flush-on-full version
    flush_wrapped = flush_on_full_cache(maxsize)(fn)

    # optional: expose same-named helpers on lru_wrapped if needed
    # e.g., lru_wrapped.cache_size = lambda: lru_wrapped.cache_info().currsize

    return {"lru": lru_wrapped, "flush_on_full": flush_wrapped, "none": fn}
