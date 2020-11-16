from discopy.biclosed import Box, Diagram
from discopy.cartesian import Function, PythonFunctor
from discopy.cat import Arrow
from discopy.rigid import PRO

from . import unification

_PYTHON_FUNCTOR = PythonFunctor(
    ob=lambda t: PRO(len(t)),
    ar=lambda f: Function(len(f.dom), len(f.cod), f.function)
)

class CallableDiagram(Diagram):
    def __call__(self, *values):
        return _PYTHON_FUNCTOR(self)(*values)

class CallableBox(Box, CallableDiagram):
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
        if self.is_dagger:
            return repr(self.dagger()) + "$^\\dagger$"
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
