
try:
    from src.utils.redis_cache import RedisCache
    cache = RedisCache(disabled=True)
except ImportError:
    # DummyCache implementation if Redis is not available
    class DummyCache:
        disabled = True
        
        def __init__(self, *args, **kwargs):
            self.disabled = True
        
        def enable(self, *args, **kwargs):
            self.disabled = False
            print("Dummy cache enabled (no actual caching will occur)")
            
        def disable(self):
            self.disabled = True
            
        def cached(self, f=None, *args, **kwargs):
            if f is None:
                from functools import partial
                return partial(self.cached, *args, **kwargs)
                
            from functools import wraps
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
            return wrapper
    
    cache = DummyCache()

import random

once_set=False
def set_seed(seed:int):
    global once_set
    if not once_set:
        once_set=True
        print(f"Setting seed to {seed}")
        random.seed(seed)