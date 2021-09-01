from discopy import messages
from discopy.biclosed import Box, Ty, Under
from discopy.cat import Arrow, Ob, AxiomError
import functools
from typing import Generic, TypeVar
import uuid

class TyVar(Ty):
    """Represents a type variable identified by a name, as a subclass of the
    :class:`discopy.biclosed.Ty` class

    :param str name: A name for the type variable
    """

    def __init__(self, name):
        super().__init__(name)

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

def unique_identifier():
    return uuid.uuid4().hex[:7]

def unique_ty():
    return Ty(unique_identifier())

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
    if a == b:
        return a, {}
    if isinstance(a, TyVar):
        return b, {a.name: b}
    if isinstance(b, TyVar):
        return a, {b.name: a}
    if a.objects and b.objects:
        results = [try_unify(ak, bk) for ak, bk in zip(a.objects, b.objects)]
        ty = Ty(*[ty for ty, _ in results])
        subst = functools.reduce(try_merge_substitution,
                                 [subst for _, subst in results])
        return ty, subst
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
