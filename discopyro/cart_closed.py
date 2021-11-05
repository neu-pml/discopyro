"""Module extending :class:`discopy.biclosed.Diagram` with a functor into
:class:`discopy.cartesian.Function` categories
"""

import discopy.biclosed as biclosed
from discopy.biclosed import Diagram, Functor
from discopy.cartesian import Function
from discopy.cat import Arrow
from discopy.rigid import PRO

from . import unification

class Box(biclosed.Box):
    """Box that supports functors into Cartesian categories of Python functions

    :param str name: Name of the box
    :param dom: Domain of the box
    :type dom: :class:`discopy.biclosed.Ty`
    :param cod: Codomain of the box
    :type cod: :class:`discopy.biclosed.Ty`

    :param function: A Python callable, or None
    :param dict data: Dictionary full of extra data to associate with the box,
                      potentially for enrichment of categories
    :param bool _dagger: Whether this box is the dagger of another
    """
    def __init__(self, name, dom, cod, function=None, data=None, _dagger=False):
        self._function = function
        super().__init__(name, dom, cod, data=data, _dagger=_dagger)

    @property
    def function(self):
        """Accessor for the Python callable contained in a box

        :return: Callable contents of the box
        :rtype: Python callable
        """
        return self._function

    def __repr__(self):
        """Represent a box's contents in string form

        :return: Text description of box
        :rtype: str
        """
        function_rep = repr(self.function) if self.function else ''
        return "Box(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), self.dom, self.cod, function_rep
        )

    def __eq__(self, other):
        """Equality predicate over callable boxes

        :param other: Another box

        :return: Whether the two boxes are equal
        :rtype: bool
        """
        if isinstance(other, Box):
            basics = all(self.__getattribute__(x) == other.__getattribute__(x)
                         for x in ['name', 'dom', 'cod', 'function'])
            subst = unification.unifier(self.typed_dom, other.typed_dom)
            subst = unification.unifier(self.typed_cod, other.typed_cod, subst)
            return basics and subst is not None
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __hash__(self):
        """Hashed string representation of a box

        :return: Hash code computed from string representation
        :rtype: str
        """
        return hash(repr(self))

class DaggerBox(Box):
    """Box with dagger that supports functors into Cartesian dagger categories
    of Python functions

    :param str name: Name of the box
    :param dom: Domain of the box
    :type dom: :class:`discopy.biclosed.Ty`
    :param cod: Codomain of the box
    :type cod: :class:`discopy.biclosed.Ty`

    :param function: A Python callable, or None
    :param dagger_function: A Python callable, or None, implementing the dagger
    :param str dagger_name: Name of the dagger box
    :param dict data: Dictionary full of extra data to associate with the box,
                      potentially for enrichment of categories
    :param bool _dagger: Whether this box is the dagger of another
    """
    def __init__(self, name, dom, cod, function=None, dagger_function=None,
                 dagger_name=None, data=None, _dagger=False):
        self._dagger_function = dagger_function
        self._dagger_name = dagger_name
        super().__init__(name, dom, cod, function, data, _dagger=_dagger)

    def dagger(self):
        """The dagger of a box

        :return: A callable box inverse/dagger to this one
        :rtype: :class:`DaggerBox`
        """
        return type(self)(
            name=self._dagger_name, dom=self.cod, cod=self.dom,
            function=self._dagger_function, dagger_function=self.function,
            dagger_name=self.name, data=self.data, _dagger=not self.is_dagger
        )

    def __repr__(self):
        """Represent a box's contents in string form, including its dagger box

        :return: Text description of box
        :rtype: str
        """
        function_rep = repr(self.function) if self.function else ''
        return "DaggerBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), self.dom, self.cod, function_rep
        )

    def __eq__(self, other):
        """Equality predicate over callable boxes, including their daggers

        :param other: Another box

        :return: Whether the two boxes are equal
        :rtype: bool
        """
        if isinstance(other, Box):
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
        """Hashed string representation of a box with its dagger

        :return: Hash code computed from string representation
        :rtype: str
        """
        return hash(repr(self))

class DaggerFunction(Function):
    """Extension of :class:`discopy.cartesian.Function` to support a dagger

    :param dom: Domain of the function
    :type dom: :class:`discopy.cartesian.Ty`
    :param cod: Codomain of the function
    :type cod: :class:`discopy.cartesian.Ty`

    :param function: A Python callable
    :param dagger_function: A Python callable implementing the dagger
    """
    def __init__(self, dom, cod, function, dagger_function):
        self._dagger_function = dagger_function
        super().__init__(dom, cod, function)

    def dagger(self):
        """Return the dagger morphism as a :class:`discopy.cartesian.Function`

        :return: Function exposing the dagger function as its callable
        :rtype: :class:`DaggerFunction`
        """
        return type(self)(self.cod, self.dom, self._dagger_function,
                          self.function)

def functionize(f):
    """Transform a :class:`Box` into a :class:`discopy.cartesian.Function`,
       including accounting for its dagger

    :param f: A box supporting a callable function, potentially with a dagger
    :type f: :class:`Box`
    """
    if isinstance(f, DaggerBox):
        dagger_function = f.dagger().function
        return DaggerFunction(len(f.dom), len(f.cod), f.function,
                              dagger_function)
    return Function(len(f.dom), len(f.cod), f.function)

_PYTHON_FUNCTOR = Functor(
    ob=lambda t: PRO(len(t)), ar=functionize,
    ob_factory=PRO, ar_factory=Function
)

Diagram.__call__ = lambda self, *values: _PYTHON_FUNCTOR(self)(*values)
