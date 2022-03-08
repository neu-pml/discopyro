from discopy import Ty
from . import cart_closed

def node_name(node):
    if isinstance(node, (cart_closed.Box, Ty)):
        return str(node)
    else:
        return 'macro[%d]' % id(node)

def data_fits_spec(data, spec):
    fits = []
    for k, v in spec.items():
        if k in data:
            fits.append(v(data[k]) if callable(v) else data[k] == v)
        else:
            fits.append(True)
    return all(fits)

class GeneratorPredicate:
    def __init__(self, path_len, min_len, path_data, cod):
        self._path_len = path_len
        self._min_len = min_len
        self._path_data = path_data
        self._cod = cod

    def __call__(self, gen, cod):
        result = True

        if self._path_data:
            fit = not isinstance(gen, cart_closed.Box) or\
                  data_fits_spec(gen.data, self._path_data)
            result = result and fit

        if self._path_len + 1 < self._min_len:
            result = result and cod != self._cod
        result = result and (cod != Ty() or self._cod == Ty())
        return result
