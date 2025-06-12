# singleton_meta.py

class SingletonMeta(type):
    """A metaclass for singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            print(f"[SingletonMeta] Creating new instance of {cls.__name__}")
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            print(f"[SingletonMeta] Returning existing instance of {cls.__name__}")
        return cls._instances[cls]
