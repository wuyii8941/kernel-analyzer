"""Bounded reuse of pure source checks, keyed by complete source text.

Not a runtime-identity cache: callers must still read the current source and
validate execution bindings. A new checker instance requires a new cache.
"""
from copy import deepcopy
from functools import lru_cache


def cached_source_checker(checker,*,max_entries=2):
    if not isinstance(max_entries,int) or isinstance(max_entries,bool) or max_entries<1:
        raise ValueError('Positive cache capacity required')
    cached=lru_cache(maxsize=max_entries)(checker)
    def check(source,symbol):
        if not isinstance(source,str) or not isinstance(symbol,str):
            raise TypeError('Complete source text and symbol required')
        # Prevent callers from changing a result subsequently returned to
        # another case. Exceptions are deliberately not cached by lru_cache.
        return deepcopy(cached(source,symbol))
    check.cache_info=cached.cache_info
    check.cache_clear=cached.cache_clear
    return check
