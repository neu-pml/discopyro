from adt import adt, Case
from discopy import messages, Ob, Ty
from discopy.cartesian import Box
from discopy.cat import AxiomError
import functools
from typing import Generic, TypeVar
import uuid

T = TypeVar('T', bound=Ob)

@adt
class Closed(Generic[T], Ob):
    BASE: Case[T]
    VAR: Case[str]
    ARROW: Case["Closed[T]", "Closed[T]"]

    def __init__(self):
        super().__init__(self._pretty(False))

    def _pretty(self, parenthesize=False):
        result = self.match(
            base=lambda ob: str(ob),
            var=lambda name: name,
            arrow=lambda l, r: '%s -> %s' % (l._pretty(True), r._pretty())
        )
        if parenthesize and self._key == Closed._Key.ARROW:
            result = '(%s)' % result
        return result

    def __len__(self):
        return self.match(
            base=len,
            var=lambda v: 1,
            arrow=lambda l, r: 1,
        )

    def __str__(self):
        return self._pretty()

class CartesianClosed(Closed[Ty]):
    def is_compound(self):
        is_arrow = self._key == Closed._Key.ARROW
        is_product = self._key == Closed._Key.BASE and len(self.base()) > 1
        return is_arrow or is_product

    def tensor(self, other):
        if self == TOP and other == TOP:
            return TOP
        if self == TOP:
            return other
        if other == TOP:
            return self
        return CartesianClosed.BASE(Ty(self, other))

    def __matmul__(self, other):
        return self.tensor(other)

TOP = CartesianClosed.BASE(Ty())

def wrap_base_ob(ob):
    if isinstance(ob, CartesianClosed):
        return ob
    if isinstance(ob, Ty):
        return CartesianClosed.BASE(ob)
    return CartesianClosed.BASE(Ty(ob))

def unique_identifier():
    return uuid.uuid4().hex[:7]

def unique_closed():
    return Closed.BASE(Ob(unique_identifier()))

UNIFICATION_EXCEPTION_MSG = 'Could not unify %s with %s'
SUBSTITUTION_EXCEPTION_MSG = 'to substitute for %s'

class UnificationException(Exception):
    def __init__(self, x, y, k=None):
        self.key = k
        self.vals = (x, y)
        if k:
            msg = UNIFICATION_EXCEPTION_MSG + ' ' + SUBSTITUTION_EXCEPTION_MSG
            self.message = msg % (x, y, k)
        else:
            self.message = UNIFICATION_EXCEPTION_MSG % (x, y)

def try_merge_substitution(lsub, rsub):
    subst = {}
    for k in {**lsub, **rsub}.keys():
        if k in lsub and k in rsub:
            _, sub = try_unify(lsub[k], rsub[k])
            subst.update(sub)
        elif k in lsub:
            subst[k] = lsub[k]
        elif k in rsub:
            subst[k] = rsub[k]
    return subst

def try_unify(a, b, subst={}):
    if isinstance(a, Ty) and isinstance(b, Ty):
        results = [try_unify(ak, bk) for ak, bk in zip(a.objects, b.objects)]
        ty = Ty(*[ty for ty, _ in results])
        subst = functools.reduce(try_merge_substitution,
                                 [subst for _, subst in results])
        return ty, subst
    if a == b:
        return a, {}
    if a._key == Closed._Key.VAR:
        return b, {a.var(): b}
    if b._key == Closed._Key.VAR:
        return a, {b.var(): a}
    if a._key == Closed._Key.ARROW and\
       b._key == Closed._Key.ARROW:
        (la, ra) = a.arrow()
        (lb, rb) = b.arrow()
        l, lsub = try_unify(la, lb)
        r, rsub = try_unify(ra, rb)
        subst = try_merge_substitution(lsub, rsub)
        return a.__class__.ARROW(l, r), subst
    raise UnificationException(a, b)

def unify(a, b, substitution={}):
    try:
        result, substitution = try_unify(a, b, substitution)
        return substitute(result, substitution)
    except UnificationException:
        return None

def unifier(a, b, substitution={}):
    try:
        _, substitution = try_unify(a, b, substitution)
        return substitution
    except UnificationException:
        return None

def substitute(t, sub):
    return t.match(
        base=Closed.BASE,
        var=lambda m: sub[m] if m in sub else Closed.VAR(m),
        arrow=lambda l, r: Closed.ARROW(substitute(l, sub), substitute(r, sub))
    )

def fold_arrow(ts):
    if len(ts) == 1:
        return ts[-1]
    return fold_arrow(ts[:-2] + [ts[-1].__class__.ARROW(ts[-2], ts[-1])])

def unfold_arrow(arrow):
    return arrow.match(
        base=lambda ob: [CartesianClosed.BASE(ob)],
        var=lambda v: [CartesianClosed.VAR(v)],
        arrow=lambda l, r: [l] + unfold_arrow(r)
    )

def fold_product(ts):
    if len(ts) == 1:
        return ts[0]
    return CartesianClosed.BASE(Ty(*ts))

def unfold_product(ty):
    return [wrap_base_ob(obj) for obj in ty.objects]

class TypedBox(Box):
    def __init__(self, name, dom, cod, function=None):
        self._type = fold_arrow([dom, cod])
        super().__init__(name, len(dom), len(cod), function)

    @property
    def type(self):
        """
        Type signature for an explicitly typed arrow between objects

        :return: A CartesianClosed for the arrow's type
        """
        return self._type

    @property
    def typed_dom(self):
        return self.type.arrow()[0]

    @property
    def typed_cod(self):
        return self.type.arrow()[1]

    def __repr__(self):
        dom, cod = self.type.arrow()
        function_rep = repr(self.function) if self.function else ''
        return "TypedBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), dom, cod, function_rep
        )

    def __eq__(self, other):
        if isinstance(other, TypedBox):
            basics = all(self.__getattribute__(x) == other.__getattribute__(x)
                         for x in ['name', 'dom', 'cod', 'function'])
            subst = unifier(self.typed_dom, other.typed_dom)
            subst = unifier(self.typed_cod, other.typed_cod, subst)
            return basics and subst is not None
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __hash__(self):
        return hash(repr(self))

class TypedDaggerBox(TypedBox):
    def __init__(self, name, dom, cod, function=None, dagger_function=None,
                 is_dagger=False):
        self._dagger_function = dagger_function
        self._dagger = is_dagger
        super().__init__(name, dom, cod, function)

    @property
    def is_dagger(self):
        return self._dagger

    def dagger(self):
        return TypedDaggerBox(self.name, self.typed_cod, self.typed_dom,
                              self._dagger_function, self._function,
                              not self._dagger)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        dom, cod = self.type.arrow()
        function_rep = repr(self.function) if self.function else ''
        return "TypedDaggerBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), dom, cod, function_rep
        )

    def __eq__(self, other):
        if isinstance(other, TypedBox):
            basics = all(self.__getattribute__(x) == other.__getattribute__(x)
                         for x in ['name', 'dom', 'cod', 'function',
                                   '_dagger_function', '_dagger'])
            subst = unifier(self.typed_dom, other.typed_dom)
            subst = unifier(self.typed_cod, other.typed_cod, subst)
            return basics and subst is not None
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __hash__(self):
        return hash(repr(self))
