"""Module supporting unification of types and type variables instantiated as
:class:`discopy.closed.Ty` instances
"""

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
    """Represents a type in nice LaTeX math formatting

    :param ty: A type to represent
    :type ty: :class:`discopy.biclosed.Ty`
    :param bool parenthesize: Whether to enclose the result in parentheses

    :return: LaTeX math formatted representation of `ty`
    :rtype: str
    """
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
    """Predicate describing whether a type is compound or not

    :param ty: A type
    :type ty: :class:`discopy.biclosed.Ty`

    :return: Whether `ty` is compound (a :class:`discopy.biclosed.Under` or a
             monoidal product type)
    :rtype: bool
    """
    return isinstance(ty, Under) or len(ty) > 1

def base_elements(ty):
    """Compute the set of primitive :class:`discopy.biclosed.Ty` elements within
       a type

    :param ty: A type
    :type ty: :class:`discopy.biclosed.Ty`

    :return: Set of `ty`'s primitive elements
    :rtype: set
    """
    if not isinstance(ty, Ty):
        return Ty(ty)
    if isinstance(ty, Under):
        return base_elements(ty.left) | base_elements(ty.right)
    bases = {ob for ob in ty.objects if not isinstance(ob, Under)}
    recursives = set().union(*[base_elements(ob) for ob in ty.objects])
    return bases | recursives

def unique_identifier():
    """Generate a universally unique identifier seven hex digits long

    :return: A seven-digit universally unique identifier in hex
    :rtype: str
    """
    return uuid.uuid4().hex[:7]

def unique_ty():
    """Generate a type with a universally unique name

    :return: A type with a universally unique name
    :rtype: :class:`discopy.biclosed.Ty`
    """
    return Ty(unique_identifier())

UNIFICATION_EXCEPTION_MSG = 'Could not unify %s with %s'
SUBSTITUTION_EXCEPTION_MSG = 'to substitute for %s'

class UnificationException(Exception):
    """Unification failure

    :param x: Type on left side of unification equation
    :type x: :class:`discopy.biclosed.Ty`
    :param y: Type on right side of unification equation
    :type y: :class:`discopy.biclosed.Ty`
    :param k: Substitution key whose resolution was subject to the equation
    :type k: str or None
    """
    def __init__(self, x, y, k=None):
        self.key = k
        self.vals = (x, y)
        if k:
            msg = UNIFICATION_EXCEPTION_MSG + ' ' + SUBSTITUTION_EXCEPTION_MSG
            self.message = msg % (x, y, k)
        else:
            self.message = UNIFICATION_EXCEPTION_MSG % (x, y)

def try_merge_substitution(lsub, rsub):
    """Try to merge two substitutions by unifying their shared variables

    :param dict lsub: Left substitution
    :param dict rsub: Right substitution

    :raises UnificationException: Failure to unify a shared variable's
                                  substituted values

    :return: A substitution enriched by unifying shared variables
    :rtype: dict
    """
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
    """Try to unify two types, potentially raising an exception with the
    incompatible components.

    :param a: Left type
    :type a: :class:`discopy.closed.Ty`
    :param b: Right type
    :type b: :class:`discopy.closed.Ty`
    :param dict subst: An initial substitution from which to fill in variables

    :raises UnificationException: Failure to unify elements of a type

    :return: A unified type and the substitution under which it was unified
    :rtype: tuple
    """
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
    if isinstance(a, Ty) and isinstance(b, Ty):
        results = [try_unify(ak, bk) for ak, bk in zip(a.objects, b.objects)]
        ty = Ty(*[ty for ty, _ in results])
        subst = functools.reduce(try_merge_substitution,
                                 [subst for _, subst in results], subst)
        return ty, subst
    raise UnificationException(a, b)

def unify(a, b, substitution={}):
    """Unify two types, returning their merger or None

    :param a: Left type
    :type a: :class:`discopy.closed.Ty`
    :param b: Right type
    :type b: :class:`discopy.closed.Ty`
    :param dict substitution: An initial substitution from which to fill in
                              variables

    :return: A unified type, or None
    :rtype: :class:`discopy.closed.Ty` or None
    """
    try:
        result, substitution = try_unify(a, b, substitution)
        return substitute(result, substitution)
    except UnificationException:
        return None

def unifier(a, b, substitution={}):
    """Unify two types, returning the substitution on success, or None

    :param a: Left type
    :type a: :class:`discopy.closed.Ty`
    :param b: Right type
    :type b: :class:`discopy.closed.Ty`
    :param dict substitution: An initial substitution from which to fill in
                              variables

    :return: A substitution, or None
    :rtype: dict or None
    """
    try:
        _, substitution = try_unify(a, b, substitution)
        return substitution
    except UnificationException:
        return None

def substitute(t, sub):
    """Substitute away the type variables in `t` under `sub`

    :param t: A type
    :type t: :class:`discopy.closed.Ty`
    :param dict sub: A substitution

    :return: A type with all variables found in `sub` substituted away
    :rtype: :class:`discopy.closed.Ty`
    """
    if isinstance(t, Under):
        return substitute(t.left, sub) >> substitute(t.right, sub)
    if isinstance(t, Ty):
        return Ty(*[substitute(ty, sub) for ty in t.objects])
    if isinstance(t, TyVar) and t.name in sub:
        return sub[t.name]
    return t

def fold_arrow(ts):
    """Combine a list of types into an arrow type

    :param list ts: A list of :class:`discopy.closed.Ty`'s

    :return: An accumulated arrow type, or the sole type in the list
    :rtype: :class:`discopy.closed.Ty`
    """
    if len(ts) == 1:
        return ts[-1]
    return fold_arrow(ts[:-2] + [ts[-2] >> ts[-1]])

def unfold_arrow(arrow):
    """Extract a list of types from an arrow type

    :param arrow: A type, preferably an arrow type
    :type arrow: :class:`discopy.closed.Ty`

    :return: A list of the arrow's components, or of the original type
    :rtype: list
    """
    if isinstance(arrow, Under):
        return [arrow.left] + unfold_arrow(arrow.right)
    return [arrow]

def fold_product(ts):
    """Combine a list of types into a product type

    :param list ts: A list of :class:`discopy.closed.Ty`'s

    :return: An accumulated product type, or the sole type in the list
    :rtype: :class:`discopy.closed.Ty`
    """
    if len(ts) == 1:
        return ts[0]
    return Ty(*ts)

def unfold_product(ty):
    """Extract a list of types from a product type

    :param ty: A type, preferably a product type
    :type ty: :class:`discopy.closed.Ty`

    :return: A list of the product's components, or of the original type
    :rtype: list
    """
    if isinstance(ty, Under):
        return [ty]
    return [Ty(ob) for ob in ty.objects]
