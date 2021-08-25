# %% imports
import unittest
from airrmap.shared.cache import Cache

class TestCache(unittest.TestCase):

    def test__init__(self):
        cache = Cache(cache_size=20)
        self.assertEqual(
            cache.cache_size, 
            20, 
            'Cache size should be set.'
        )

    def test_count(self):
        cache = Cache()
        cache.store('k1', 'item1')
        cache.store('k2', 'item2')
        self.assertEqual(
            cache.count(), 
            2, 
            'count() should return the number of items in the cache.'
        )

    def test_store(self):
        cache = Cache(cache_size=2)
        cache.store('k1', 'item1')
        cache.store('k2', 'item2')
        cache.store('k3', 'item3')
        self.assertEqual(
            cache.first(),
            'item2',
            'store() should drop the oldest item if cache size is exceeded.'
        )
        cache.store('k2', 'item2')
        self.assertEqual(
            cache.last(),
            'item2',
            'store() with existing key should remove the old item and ' +
            'store the new item as the last item in the cache.'
        )


    def test_get(self):
        cache = Cache()
        cache.store('k1', 'item1')
        cache.store('k2', 'item2')
        item2 = cache.get('k2')
        self.assertEqual(
            item2,
            'item2',
            'get() should return the requested item if key exists.'
        )

        item_none = cache.get('does_not_exist', if_not_exists='xx')
        self.assertEqual(
            item_none,
            'xx',
            'get() should return if_not_exists if key is not found.'
        )

    def test_first(self):
        cache = Cache()
        cache.store('k1', 'item1')
        cache.store('k2', 'item2')
        self.assertEqual(
            cache.first(),
            'item1',
            'first() should return the oldest stored item.'
        )
        
        cache.clear()
        self.assertEqual(
            cache.first(if_empty='xx'),
            'xx',
            'first() should return if_empty if the cache is empty.'
        )

    def test_last(self):
        cache = Cache()
        cache.store('k1', 'item1')
        cache.store('k2', 'item2')
        self.assertEqual(
            cache.last(),
            'item2',
            'last() should return the most recently stored item.'
        )
        
        cache.clear()
        self.assertEqual(
            cache.last(if_empty='xx'),
            'xx',
            'last() should return if_empty if the cache is empty.'
        )
        

    def test_clear(self):
        cache = Cache()
        cache.store('k1', 'item1')
        cache.clear()
        self.assertEqual(
            cache.count(),
            0,
            'clear() should remove all items from the cache.'
        )

# %%
if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
