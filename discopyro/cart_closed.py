"""Module extending :class:`discopy.biclosed.Diagram` with a functor into
:class:`discopy.python.Function` categories
"""

from discopy.cartesian import Box
from discopy.python import Function
from discopy.cat import Arrow
from discopy.rigid import PRO

from . import unification

class DaggerBox(Box):
    """Box with dagger that supports functors into Cartesian dagger categories

    :param str name: Name of the box
    :param dom: Domain of the box
    :type dom: :class:`discopy.cartesian.Ty`
    :param cod: Codomain of the box
    :type cod: :class:`discopy.cartesian.Ty`

    :param function: A Python callable, or None
    :param dagger_function: A Python callable, or None, implementing the dagger
    :param str dagger_name: Name of the dagger box
    :param dict data: Dictionary full of extra data to associate with the box,
                      potentially for enrichment of categories
    :param bool _dagger: Whether this box is the dagger of another
    """
    def __init__(self, name, dom, cod, dagger_name=None, _dagger=False,
                 data=None):
        self._dagger_name = dagger_name
        super().__init__(name, dom, cod, data, _dagger=_dagger)

    def dagger(self):
        """The dagger of a box

        :return: A callable box inverse/dagger to this one
        :rtype: :class:`DaggerBox`
        """
        return type(self)(
            name=self._dagger_name, dom=self.cod, cod=self.dom,
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
                         for x in ['name', 'dom', 'cod', '_dagger_name',
                                   '_dagger'])
            subst = unification.unifier(self.dom, other.dom)
            subst = unification.unifier(self.cod, other.cod, subst)
            return basics and subst is not None
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

class DaggerFunction(Function):
    """Extension of :class:`discopy.python.Function` to support a dagger

    :param dom: Domain of the function
    :type dom: :class:`discopy.python.Ty`
    :param cod: Codomain of the function
    :type cod: :class:`discopy.python.Ty`

    :param function: A Python callable
    :param dagger_function: A Python callable implementing the dagger
    """
    def __init__(self, function, dom, cod, dagger_function):
        self._dagger_function = dagger_function
        super().__init__(function, dom, cod)

    def dagger(self):
        """Return the dagger morphism as a :class:`discopy.python.Function`

        :return: Function exposing the dagger function as its callable
        :rtype: :class:`DaggerFunction`
        """
        return type(self)(self._dagger_function, self.cod, self.dom,
                          self.inside)

def functionize(f):
    """Transform a :class:`Box` into a :class:`discopy.python.Function`,
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
