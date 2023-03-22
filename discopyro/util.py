from discopy.monoidal import Box, Ty

def type_contains(tx, ty):
    if len(ty) < len(tx):
        return False
    for i, ob in enumerate(ty.inside):
        if ob == tx[0]:
            for j, x in enumerate(tx.inside):
                if i + j >= len(ty) or ty[i+j] != x:
                    return False
            return True
    return False

def node_name(node):
    if isinstance(node, (Box, Ty)):
        return str(node)
    return 'macro[%d]' % id(node)

def data_fits_spec(data, spec):
    fits = []
    for k, v in spec.items():
        if k in data:
            fits.append(v(data[k]) if callable(v) else data[k] == v)
        else:
            fits.append(True)
    return all(fits)

class HomsetPredicate:
    def __init__(self, path_len, min_len, cod):
        self._path_len = path_len
        self._min_len = min_len
        self._cod = cod

    @property
    def cod(self):
        return self._cod

    def __call__(self, edge):
        dom, cod = edge
        result = True
        if self._path_len + 1 < self._min_len:
            result = result and cod != self._cod
        result = result and cod != Ty()

        return result

class GeneratorPredicate:
    def __init__(self, path_data):
        self._path_data = path_data

    def __call__(self, gen):
        result = True

        if self._path_data:
            fit = not isinstance(gen, Box) or data_fits_spec(gen.data,
                                                             self._path_data)
            result = result and fit

        return result
