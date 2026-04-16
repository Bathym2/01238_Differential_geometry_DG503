import sympy as sp

from .functions import (
    calculate_christoffel_symbols,
    calculate_g_dot,
    calculate_g_norm,
)


def _make_point_subs(pt, coords):
    if isinstance(pt, dict):
        return pt
    if isinstance(pt, (list, tuple, sp.Matrix)):
        if len(pt) != len(coords):
            raise ValueError("Point coordinate list length must match coords length.")
        return {coords[i]: pt[i] for i in range(len(coords))}
    raise TypeError("pt must be a dict or list/tuple/Matrix of coordinate values.")


def calculate_curvature_operator(g_matrix: sp.Matrix, coords: list) -> dict:
    """
    Calculates the curvature operator components R^m_{ijk} for a metric g.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A dict mapping index tuples (i, j, k, m) to sympy expressions.
    """
    dim = len(coords)
    chris = calculate_christoffel_symbols(g_matrix, coords)
    Rop = {}

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    val = (
                        sp.diff(chris[(j, k, m)], coords[i])
                        - sp.diff(chris[(i, k, m)], coords[j])
                        + sum(
                            chris[(i, s, m)] * chris[(j, k, s)]
                            - chris[(j, s, m)] * chris[(i, k, s)]
                            for s in range(dim)
                        )
                    )
                    Rop[(i, j, k, m)] = sp.simplify(val)
    return Rop


def calculate_curvature_tensor(g_matrix: sp.Matrix, coords: list) -> dict:
    """
    Calculates the fully lowered curvature tensor R_{ijkm} from the curvature operator.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A dict mapping index tuples (i, j, k, m) to sympy expressions.
    """
    dim = len(coords)
    Rop = calculate_curvature_operator(g_matrix, coords)
    Rten = {}

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m in range(dim):
                    val = sum(
                        Rop[(i, j, k, s)] * g_matrix[s, m]
                        for s in range(dim)
                    )
                    Rten[(i, j, k, m)] = sp.simplify(val)
    return Rten


def calculate_sectional_curvature(
    g_matrix: sp.Matrix,
    pt,
    X_vec,
    Y_vec,
    coords: list,
) -> sp.Expr:
    """
    Calculates sectional curvature of the plane spanned by X and Y at a point.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        pt: Point as dict {x1: a, x2: b, ...} or list/tuple/Matrix of values.
        X_vec: Vector components [X^1, ..., X^n].
        Y_vec: Vector components [Y^1, ..., Y^n].
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        The sectional curvature scalar at the point.
    """
    subs_pt = _make_point_subs(pt, coords)
    X = sp.Matrix(X_vec)
    Y = sp.Matrix(Y_vec)
    g_at_pt = g_matrix.subs(subs_pt)
    Rten = calculate_curvature_tensor(g_matrix, coords)

    numerator = sum(
        Rten[(i, j, k, m)].subs(subs_pt) * X[i] * Y[j] * Y[k] * X[m]
        for i in range(len(coords))
        for j in range(len(coords))
        for k in range(len(coords))
        for m in range(len(coords))
    )
    denominator = (
        calculate_g_dot(g_at_pt, X, X)
        * calculate_g_dot(g_at_pt, Y, Y)
        - calculate_g_dot(g_at_pt, X, Y) ** 2
    )
    return sp.simplify(numerator / denominator)


def calculate_ricci_tensor_on_vectors(
    g_matrix: sp.Matrix,
    pt,
    X_vec,
    Y_vec,
    coords: list,
) -> sp.Expr:
    """
    Calculates the Ricci tensor evaluated on vectors X and Y at a point.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        pt: Point as dict {x1: a, x2: b, ...} or list/tuple/Matrix of values.
        X_vec: Vector components [X^1, ..., X^n].
        Y_vec: Vector components [Y^1, ..., Y^n].
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        The Ricci curvature scalar Ric(X, Y) at the point.
    """
    subs_pt = _make_point_subs(pt, coords)
    X = sp.Matrix(X_vec)
    Y = sp.Matrix(Y_vec)
    g_at_pt = g_matrix.subs(subs_pt)
    g_inv_at_pt = g_at_pt.inv()
    Rten = calculate_curvature_tensor(g_matrix, coords)

    value = sum(
        Rten[(i, j, k, m)].subs(subs_pt)
        * g_inv_at_pt[j, k]
        * X[i]
        * Y[m]
        for i in range(len(coords))
        for j in range(len(coords))
        for k in range(len(coords))
        for m in range(len(coords))
    )
    return sp.simplify(value)


def calculate_ricci_curvature(
    g_matrix: sp.Matrix,
    pt,
    X_vec,
    coords: list,
) -> sp.Expr:
    """
    Calculates the Ricci curvature in the direction of X at a point.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        pt: Point as dict {x1: a, x2: b, ...} or list/tuple/Matrix of values.
        X_vec: Vector components [X^1, ..., X^n].
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        The Ricci curvature scalar in direction X at the point.
    """
    subs_pt = _make_point_subs(pt, coords)
    X = sp.Matrix(X_vec)
    g_at_pt = g_matrix.subs(subs_pt)
    eX = X / calculate_g_norm(g_at_pt, X)
    return calculate_ricci_tensor_on_vectors(g_matrix, pt, eX, eX, coords)


def calculate_scalar_curvature(
    g_matrix: sp.Matrix,
    pt,
    coords: list,
) -> sp.Expr:
    """
    Calculates the scalar curvature at a point.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        pt: Point as dict {x1: a, x2: b, ...} or list/tuple/Matrix of values.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        The scalar curvature at the point.
    """
    subs_pt = _make_point_subs(pt, coords)
    g_at_pt = g_matrix.subs(subs_pt)
    g_inv_at_pt = g_at_pt.inv()
    Rten = calculate_curvature_tensor(g_matrix, coords)

    scalar = sum(
        Rten[(i, j, k, m)].subs(subs_pt)
        * g_inv_at_pt[i, m]
        * g_inv_at_pt[j, k]
        for i in range(len(coords))
        for j in range(len(coords))
        for k in range(len(coords))
        for m in range(len(coords))
    )
    return sp.simplify(scalar)
