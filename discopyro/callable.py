from discopy.biclosed import Box, Diagram, Functor
from discopy.cartesian import Function
from discopy.cat import Arrow
from discopy.rigid import PRO

from . import unification

class CallableBox(Box):
    def __init__(self, name, dom, cod, function=None, data=None, _dagger=False):
        self._function = function
        super().__init__(name, dom, cod, data, _dagger=_dagger)

    @property
    def function(self):
        return self._function

    def __repr__(self):
        function_rep = repr(self.function) if self.function else ''
        return "CallableBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), self.dom, self.cod, function_rep
        )

    def __eq__(self, other):
        if isinstance(other, CallableBox):
            basics = all(self.__getattribute__(x) == other.__getattribute__(x)
                         for x in ['name', 'dom', 'cod', 'function'])
            subst = unification.unifier(self.typed_dom, other.typed_dom)
            subst = unification.unifier(self.typed_cod, other.typed_cod, subst)
            return basics and subst is not None
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __hash__(self):
        return hash(repr(self))

class CallableDaggerBox(CallableBox):
    def __init__(self, name, dom, cod, function=None, dagger_function=None,
                 dagger_name=None, data=None, _dagger=False):
        self._dagger_function = dagger_function
        self._dagger_name = dagger_name
        super().__init__(name, dom, cod, function, data, _dagger=_dagger)

    def dagger(self):
        return type(self)(
            name=self._dagger_name, dom=self.cod, cod=self.dom,
            function=self._dagger_function, dagger_function=self.function,
            dagger_name=self.name, data=self.data, _dagger=not self.is_dagger
        )

    def __repr__(self):
        function_rep = repr(self.function) if self.function else ''
        return "CallableDaggerBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), self.dom, self.cod, function_rep
        )

    def __eq__(self, other):
        if isinstance(other, CallableBox):
            basics = all(self.__getattribute__(x) == other.__getattribute__(x)
                         for x in ['name', 'dom', 'cod', 'function',
                                   '_dagger_function', '_dagger_name',
                                   '_dagger'])
            subst = unification.unifier(self.dom, other.dom)
            subst = unification.unifier(self.cod, other.cod, subst)
            return basics and subst is not None
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __hash__(self):
        return hash(repr(self))

class DaggerFunction(Function):
    def __init__(self, dom, cod, function, dagger_function):
        self._dagger_function = dagger_function
        super().__init__(dom, cod, function)

    def dagger(self):
        return type(self)(self.cod, self.dom, self._dagger_function,
                          self.function)

def functionize(f):
    if isinstance(f, CallableDaggerBox):
        dagger_function = f.dagger().function
        return DaggerFunction(len(f.dom), len(f.cod), f.function,
                              dagger_function)
    return Function(len(f.dom), len(f.cod), f.function)

_PYTHON_FUNCTOR = Functor(
    ob=lambda t: PRO(len(t)), ar=functionize,
    ob_factory=PRO, ar_factory=Function
)

Diagram.__call__ = lambda self, *values: _PYTHON_FUNCTOR(self)(*values)
