from cachetools import TTLCache
from functools import wraps
import hashlib
import json
from utils.logger import logger


# In-memory cache (5 minutes TTL, max 100 items)
cache = TTLCache(maxsize=100, ttl=300)


def cache_key(*args, **kwargs):
    """Generate cache key from arguments"""
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl=300):
    """
    Decorator to cache function results
    Usage: @cached(ttl=300)  # Cache for 5 minutes
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{cache_key(*args, **kwargs)}"
            
            # Check cache
            if key in cache:
                logger.debug(f"Cache HIT: {func.__name__}")
                return cache[key]
            
            # Execute function
            logger.debug(f"Cache MISS: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            
            return result
        return wrapper
    return decorator


def clear_cache():
    """Clear all cached data"""
    cache.clear()
    logger.info("Cache cleared")

