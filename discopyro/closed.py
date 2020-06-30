from adt import adt, Case
from discopy import messages, Ob, Ty
from discopy.cartesian import Function, tuplify
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

class CartesianClosed(Closed[Ty]):
    def is_compound(self):
        is_arrow = self._key == Closed._Key.ARROW
        is_product = self._key == Closed._Key.BASE and len(self.base()) > 1
        return is_arrow or is_product

TOP = CartesianClosed.BASE(Ty())

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
        return Closed.ARROW(l, r), subst
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
    return fold_arrow(ts[:-2] + [Closed.ARROW(ts[-2], ts[-1])])

def unfold_arrow(arrow):
    return arrow.match(
        base=lambda ob: [CartesianClosed.BASE(ob)],
        var=lambda v: [CartesianClosed.VAR(v)],
        arrow=lambda l, r: [l] + unfold_arrow(r)
    )

class TypedFunction(Function):
    def __init__(self, dom, cod, function):
        # Deconstruct the type here into dom, cod, and self.forward
        self._type = fold_arrow([dom, cod])
        super().__init__(len(dom), len(cod), function)

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
        return "TypedFunction(dom={}, cod={}, function={})".format(
            dom, cod, repr(self.function)
        )

    def then(self, other):
        if isinstance(other, TypedFunction):
            tx = self.type
            dom, midx = tx.arrow()
            ty = other.type
            midy, cod = ty.arrow()
            subst = unifier(midx, midy)
            if subst is None:
                raise AxiomError(messages.does_not_compose(self, other))
            return TypedFunction(substitute(dom, subst), substitute(cod, subst),
                                 lambda *vals: other(*tuplify(self(*vals))))
        return super().then(other)
