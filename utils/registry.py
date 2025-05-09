# utils/registry.py
registry = dict()

def register(name):
    def decorator(cls):
        registry[name.lower()] = cls   # 强制小写键
        return cls
    return decorator

def get(name):
    return registry[name.lower()]     # 强制小写查找