from adt import adt, Case
from discopy import messages
from discopy.biclosed import Box, Ty, Under
from discopy.cat import Arrow, Ob, AxiomError
import functools
from typing import Generic, TypeVar
import uuid

class TyVar(Ty):
    def __init__(self, name):
        super().__init__(name)

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
            base=str,
            var=lambda name: 'Var(%s)' % name,
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

def pretty_tuple_type(ty):
    if not ty.objects:
        return '\\top'
    return ' \\times '.join([str(obj) for obj in ty.objects])

def pretty_type(ty, parenthesize=False):
    if isinstance(ty, Under):
        result = '%s >> %s' % (pretty_type(ty.left, True),
                               pretty_type(ty.right))
        if parenthesize:
            result = '(%s)' % result
    else:
        if not ty.objects:
            result = '\\top'
        result = ' \\times '.join([str(obj) for obj in ty.objects])
    return result

def type_compound(ty):
    return isinstance(ty, Under) or len(ty) > 1

def base_elements(ty):
    if not isinstance(ty, Ty):
        return Ty(ty)
    if isinstance(ty, Under):
        return base_elements(ty.left) | base_elements(ty.right)
    bases = {ob for ob in ty.objects if not isinstance(ob, Under)}
    recursives = set().union(*[base_elements(ob) for ob in ty.objects])
    return bases | recursives

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

    def _pretty(self, parenthesize=False):
        result = self.match(
            base=lambda ty: pretty_tuple_type(ty),
            var=lambda name: 'Var(%s)' % name,
            arrow=lambda l, r: '%s -> %s' % (l._pretty(True), r._pretty())
        )
        if parenthesize and self._key == Closed._Key.ARROW:
            result = '(%s)' % result
        return result

    def base_elements(self):
        def ty_base_elements(ty):
            return set().union(*[t.base_elements() for t in ty
                                 if isinstance(t, CartesianClosed)]) |\
                   {t for t in ty if not isinstance(t, CartesianClosed)}
        return self.match(
            base=ty_base_elements,
            var=lambda name: set(CartesianClosed.VAR(name)),
            arrow=lambda l, r: l.base_elements() | r.base_elements()
        )

TOP = Ty()

def wrap_base_ob(ob):
    if isinstance(ob, CartesianClosed):
        return ob
    if isinstance(ob, Ty):
        return CartesianClosed.BASE(ob)
    return CartesianClosed.BASE(Ty(ob))

def unique_identifier():
    return uuid.uuid4().hex[:7]

def unique_ty():
    return Ty(unique_identifier())

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
    if isinstance(a, Under) and isinstance(b, Under):
        l, lsub = try_unify(a.left, b.left)
        r, rsub = try_unify(a.right, b.right)
        subst = try_merge_substitution(lsub, rsub)
        return l >> r, subst
    if a.objects and b.objects:
        results = [try_unify(ak, bk) for ak, bk in zip(a.objects, b.objects)]
        ty = Ty(*[ty for ty, _ in results])
        subst = functools.reduce(try_merge_substitution,
                                 [subst for _, subst in results])
        return ty, subst
    if a == b:
        return a, {}
    if isinstance(a, TyVar):
        return b, {a.name: b}
    if isinstance(b, TyVar):
        return a, {b.name: a}
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
    if isinstance(t, Under):
        return substitute(t.left, sub) >> substitute(t.right, sub)
    if t.objects:
        return Ty(*[substitute(ty, sub) for ty in t.objects])
    if isinstance(t, TyVar) and t.name in sub:
        return sub[t.name]
    return t

def fold_arrow(ts):
    if len(ts) == 1:
        return ts[-1]
    return fold_arrow(ts[:-2] + [ts[-2] >> ts[-1]])

def unfold_arrow(arrow):
    if isinstance(arrow, Under):
        return [arrow.left] + unfold_arrow(arrow.right)
    return [arrow]

def fold_product(ts):
    if len(ts) == 1:
        return ts[0]
    return Ty(*ts)

def unfold_product(ty):
    if isinstance(ty, Under):
        return [ty]
    return [Ty(ob) for ob in ty.objects]

class TypedBox(Box):
    def __init__(self, name, dom, cod, function=None):
        super().__init__(name, dom, cod, function)

    @property
    def type(self):
        """
        Type signature for an explicitly typed arrow between objects

        :return: A CartesianClosed for the arrow's type
        """
        return self.dom >> self.cod

    @property
    def typed_dom(self):
        return self.dom

    @property
    def typed_cod(self):
        return self.cod

    def __repr__(self):
        function_rep = repr(self.function) if self.function else ''
        return "TypedBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), self.dom, self.cod, function_rep
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
                 dagger_name=None, is_dagger=False):
        self._dagger_function = dagger_function
        self._dagger_name = dagger_name
        self._dagger = is_dagger
        super().__init__(name, dom, cod, function)

    @property
    def is_dagger(self):
        return self._dagger

    def dagger(self):
        if self._dagger_name:
            dagger_name = self._dagger_name
        else:
            dagger_name = self.name + "$^\\dagger$"
        return TypedDaggerBox(dagger_name, self.typed_cod, self.typed_dom,
                              self._dagger_function, self._function, self.name,
                              not self._dagger)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + "$^\\dagger$"
        function_rep = repr(self.function) if self.function else ''
        return "TypedDaggerBox(name={}, dom={}, cod={}, function={})".format(
            repr(self.name), self.dom, self.cod, function_rep
        )

    def __eq__(self, other):
        if isinstance(other, TypedBox):
            basics = all(self.__getattribute__(x) == other.__getattribute__(x)
                         for x in ['name', 'dom', 'cod', 'function',
                                   '_dagger_function', '_dagger_name',
                                   '_dagger'])
            subst = unifier(self.typed_dom, other.typed_dom)
            subst = unifier(self.typed_cod, other.typed_cod, subst)
            return basics and subst is not None
        elif isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __hash__(self):
        return hash(repr(self))
