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

class HomsetPredicate:
    def __init__(self, path_len, min_len, cod, generators={}, path_data={}):
        self._path_len = path_len
        self._min_len = min_len
        self._cod = cod
        self._generators = generators
        self._path_data = path_data

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
