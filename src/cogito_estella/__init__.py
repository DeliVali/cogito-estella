__version__ = "0.19.1"


def __getattr__(name):
    # lazy: `import cogito_estella` stays free; `from cogito_estella import Estella` loads it
    if name == "Estella":
        from cogito_estella.estella import Estella
        return Estella
    raise AttributeError(name)
