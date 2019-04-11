from functools import wraps

#decorator to kill cache
def cache_clear(f):
    def wrapper(*args):
        self = args[0]
        self._clear_cache()
        return f(*args)
    return wrapper


#decorator to cache property
def cache_prop(fn):
    key = fn.__name__
    @property
    def wrapper(self):
        if self._cached[key] is None:
            #print('generating',key)
            self._cached[key] = fn(self)
        return self._cached[key]
    return wrapper


#decorator to cache function
def cache_func(fn):
    key0 = fn.__name__
    @wraps(fn)
    def wrapper(*args):
        self = args[0]
        key = (key0,) + args[1:]
        if self._cached[key] is None:
            #print('generating',key)
            self._cached[key] = fn(*args)
        return self._cached[key]
    return wrapper
