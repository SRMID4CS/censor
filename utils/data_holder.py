# data_holder.py

from utils.singleton_meta import SingletonMeta

class DataHolder(metaclass=SingletonMeta):
    def __init__(self):
        self._data = {}

    def set(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def all_data(self):
        return self._data.copy()
