
# Simple cache class

# tests > test_cache.py

from collections import OrderedDict
from typing import Any

class Cache():

    def __init__(self, cache_size = 100):

        self._cache = OrderedDict()
        self.cache_size = cache_size    

    def count(self):
        """Return the number of items in the cache"""
        return len(self._cache)
        
    def store(self, key: Any, item: Any):
        """
        Store an item in the cache.

        If the cache size is exceeded, the
        oldest item will be removed.
        
        Parameters
        ----------
        key : Any
            The key for the item. If an item
            already exists with the same key,
            the item will be removed first.
            Not limited to strings - can be 
            any key compatible with OrderedDict.

        item : Any
            The item to store in the cache.
            It will be added to the end of the cache.
        """
        
        # Remove previous if existing
        if key in self._cache:
            del self._cache[key]

        # Store
        self._cache[key] = item

        # Remove oldest/first item if over the cache size
        if len(self._cache) > self.cache_size:
            oldest_item = self._cache.popitem(last=False)


    def get(self, key: Any, if_not_exists: Any=None)->Any:
        """
        Retrieve an item from the cache.

        Parameters
        ----------
        key : Any
            Key for the item.
        if_not_exists : Any, optional
            The value to return if the key doesn't exist, by default None.

        Returns
        -------
        Any
            The item in the cache if key exists, otherwise if_not_exists.
        """
    
        if key in self._cache:
            return self._cache[key]
        else:
            return if_not_exists


        
    def first(self, if_empty: Any=None)->Any:
        """
        Return the first item in the cache.

        Parameters
        ----------
        if_empty : Any, optional
            Value to return if the cache is empty, by default None.

        Returns
        -------
        Any
            The first item or if_empty if the cache is empty.
        """
        if len(self._cache) == 0:
            return if_empty
        else:
            return self._cache[next(iter(self._cache))]

    
    def last(self, if_empty: Any=None)->Any:
        """
        Return the last item in the cache.

        Parameters
        ----------
        if_empty : Any, optional
            Value to return if the cache is empty, by default None.

        Returns
        -------
        Any
            The last item or if_empty if the cache is empty.
        """
        if len(self._cache) == 0:
            return if_empty
        else:
            return self._cache[next(reversed(self._cache))]
        
        

    def clear(self):
        """Clear the cache"""
        self._cache.clear()
    

