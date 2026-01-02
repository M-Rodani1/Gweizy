from cachetools import TTLCache
from functools import wraps
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional
from utils.logger import logger
import threading


# Enhanced in-memory cache with better tracking
cache = TTLCache(maxsize=500, ttl=300)  # Increased size
cache_stats: Dict[str, Dict[str, int]] = {
    'hits': {},
    'misses': {},
    'sets': {}
}
_cache_lock = threading.Lock()


def cache_key(*args, **kwargs):
    """Generate cache key from arguments"""
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def _update_stats(func_name: str, stat_type: str):
    """Update cache statistics"""
    with _cache_lock:
        if func_name not in cache_stats[stat_type]:
            cache_stats[stat_type][func_name] = 0
        cache_stats[stat_type][func_name] += 1


def cached(ttl=300, key_prefix: Optional[str] = None):
    """
    Enhanced decorator to cache function results with statistics
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Optional prefix for cache key
    
    Usage: 
        @cached(ttl=300, key_prefix='predictions')
        def get_predictions():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prefix = key_prefix or func.__name__
            key = f"{prefix}_{cache_key(*args, **kwargs)}"
            
            # Check cache
            if key in cache:
                _update_stats(func.__name__, 'hits')
                logger.debug(f"Cache HIT: {func.__name__} (key: {key[:20]}...)")
                return cache[key]
            
            # Execute function
            _update_stats(func.__name__, 'misses')
            logger.debug(f"Cache MISS: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            _update_stats(func.__name__, 'sets')
            
            return result
        return wrapper
    return decorator


def clear_cache(pattern: Optional[str] = None):
    """
    Clear cached data
    
    Args:
        pattern: Optional pattern to match keys (if None, clears all)
    """
    if pattern:
        keys_to_remove = [k for k in cache.keys() if pattern in k]
        for key in keys_to_remove:
            del cache[key]
        logger.info(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")
    else:
        cache.clear()
        logger.info("Cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    with _cache_lock:
        total_hits = sum(cache_stats['hits'].values())
        total_misses = sum(cache_stats['misses'].values())
        total_requests = total_hits + total_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(cache),
            'cache_maxsize': cache.maxsize,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'by_function': {
                func: {
                    'hits': cache_stats['hits'].get(func, 0),
                    'misses': cache_stats['misses'].get(func, 0),
                    'hit_rate': round(
                        (cache_stats['hits'].get(func, 0) / 
                         (cache_stats['hits'].get(func, 0) + cache_stats['misses'].get(func, 0)) * 100)
                        if (cache_stats['hits'].get(func, 0) + cache_stats['misses'].get(func, 0)) > 0 else 0,
                        2
                    )
                }
                for func in set(list(cache_stats['hits'].keys()) + list(cache_stats['misses'].keys()))
            }
        }


def warm_cache(func, *args, **kwargs):
    """
    Pre-warm cache by calling a function
    
    Args:
        func: Function to call
        *args, **kwargs: Arguments to pass to function
    """
    try:
        result = func(*args, **kwargs)
        logger.info(f"Cache warmed for {func.__name__}")
        return result
    except Exception as e:
        logger.warning(f"Failed to warm cache for {func.__name__}: {e}")
        return None

