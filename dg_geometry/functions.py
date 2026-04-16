import sympy as sp

def calculate_jacobi_function(r_vec: sp.Matrix, params: list) -> sp.Expr:
    """
    Calculates the absolute valued Jacobi function for a vector function r_vec.
    Generalizes arc length element (1D), surface area element (2D), and volume element (3D).

    Args:
        r_vec: A sympy.Matrix representing the vector function in R^3.
        params: A list of sympy.Symbol objects [u], [u, v], or [u, v, w].

    Returns:
        A sympy expression representing the absolute Jacobi function.
    """
    num_params = len(params)

    if num_params == 1:
        # Case 1: Curve speed ||dr/du||
        u = params[0]
        r_prime = sp.diff(r_vec, u)
        jacobi = r_prime.norm()

    elif num_params == 2:
        # Case 2: Surface area element ||dr/du x dr/dv||
        u, v = params
        r_u = sp.diff(r_vec, u)
        r_v = sp.diff(r_vec, v)
        cross_product = r_u.cross(r_v)
        jacobi = cross_product.norm()

    elif num_params == 3:
        # Case 3: Volume element |det(J)|
        u, v, w = params
        r_u = sp.diff(r_vec, u)
        r_v = sp.diff(r_vec, v)
        r_w = sp.diff(r_vec, w)
        jacobian_matrix = r_u.row_join(r_v).row_join(r_w)
        jacobi = sp.Abs(jacobian_matrix.det())

    else:
        raise ValueError("Unsupported number of parameters. This function supports 1, 2, or 3 parameters.")

    return sp.simplify(jacobi)

def calculate_signed_jacobi_function(r_vec: sp.Matrix, params: list) -> sp.Expr:
    """
    Calculates the signed Jacobi function for a vector function r_vec.
    Matches logic where 1D and 2D return magnitudes, and 3D returns the determinant.

    Args:
        r_vec: A sympy.Matrix representing the vector function in R^3.
        params: A list of sympy.Symbol objects [u], [u, v], or [u, v, w].

    Returns:
        A sympy expression representing the signed Jacobi function.
    """
    num_params = len(params)

    if num_params == 1:
        u = params[0]
        r_prime = sp.diff(r_vec, u)
        jacobi = r_prime.norm()

    elif num_params == 2:
        u, v = params
        r_u = sp.diff(r_vec, u)
        r_v = sp.diff(r_vec, v)
        cross_product = r_u.cross(r_v)
        jacobi = cross_product.norm()

    elif num_params == 3:
        u, v, w = params
        r_u = sp.diff(r_vec, u)
        r_v = sp.diff(r_vec, v)
        r_w = sp.diff(r_vec, w)
        jacobian_matrix = r_u.row_join(r_v).row_join(r_w)
        jacobi = jacobian_matrix.det()

    else:
        raise ValueError("Unsupported number of parameters. This function supports 1, 2, or 3 parameters.")

    return sp.simplify(jacobi)

def calculate_g_dot(g_matrix: sp.Matrix, V: sp.Matrix, W: sp.Matrix) -> sp.Expr:
    """
    Calculates the dot product of two vectors V and W with respect to the metric g.
    Matches DG503[Gdot] logic (V^T * g * W).
    """
    return sp.simplify((V.T * g_matrix * W)[0, 0])

def calculate_g_norm(g_matrix: sp.Matrix, V: sp.Matrix) -> sp.Expr:
    """
    Calculates the norm of a vector V with respect to the metric g.
    Matches DG503[Gnrm] logic (sqrt(V^T * g * V)).
    """
    return sp.simplify(sp.sqrt((V.T * g_matrix * V)[0, 0]))

def calculate_lie_derivative_metric(g_matrix: sp.Matrix, X_vec: sp.Matrix, coords: list) -> sp.Matrix:
    """
    Calculates the Lie derivative of a metric matrix g with respect to a vector field X.
    Matches DG503[LieG] logic.
    L_X g_ij = X^k d_k g_ij + g_jk d_i X^k + g_ik d_j X^k
    """
    dim = len(coords)
    lie_g = sp.zeros(dim, dim)
    for i in range(dim):
        for j in range(dim):
            val = sum(
                X_vec[k] * sp.diff(g_matrix[i, j], coords[k]) +
                g_matrix[j, k] * sp.diff(X_vec[k], coords[i]) +
                g_matrix[i, k] * sp.diff(X_vec[k], coords[j])
                for k in range(dim)
            )
            lie_g[i, j] = sp.simplify(val)
    return lie_g

def calculate_lie_bracket(X_vec: sp.Matrix, Y_vec: sp.Matrix, coords: list) -> sp.Matrix:
    """
    Calculates the Lie bracket [X, Y] of two vector fields.
    Matches DG503[LieVec] logic.
    [X, Y]^i = X^j d_j Y^i - Y^j d_j X^i
    """
    dim = len(coords)
    res = sp.zeros(dim, 1)
    for i in range(dim):
        val = sum(
            X_vec[j] * sp.diff(Y_vec[i], coords[j]) -
            Y_vec[j] * sp.diff(X_vec[i], coords[j])
            for j in range(dim)
        )
        res[i] = sp.simplify(val)
    return res

def calculate_christoffel_symbols(g_matrix: sp.Matrix, coords: list) -> dict:
    """
    Calculates the Christoffel symbols of the second kind for a metric matrix g.
    Matches DG503[ChristoffelAll] logic.
    Gamma^k_ij = 1/2 * sum_m ( g^km * (d_i g_jm + d_j g_im - d_m g_ij) )

    Args:
        g_matrix: The metric tensor as a sympy Matrix.
        coords: List of coordinate symbols (e.g., [u, v] or [x1, x2, x3]).

    Returns:
        A dictionary with keys (i, j, k) corresponding to Gamma^k_ij.
    """
    dim = len(coords)
    g_inv = g_matrix.inv()
    chris = {}
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                val = sum(
                    g_inv[k, m] * (
                        sp.diff(g_matrix[j, m], coords[i]) +
                        sp.diff(g_matrix[i, m], coords[j]) -
                        sp.diff(g_matrix[i, j], coords[m])
                    )
                    for m in range(dim)
                )
                chris[(i, j, k)] = sp.simplify(sp.Rational(1, 2) * val)
    return chris


def calculate_covariant_derivative_along_curve(
    g_matrix: sp.Matrix,
    gamma_vec: sp.Matrix,
    V_vec: sp.Matrix,
    t: sp.Symbol,
    coords: list,
) -> sp.Matrix:
    """
    Calculates the covariant derivative of a vector field V along a curve gamma(t).

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        gamma_vec: Curve vector [x1(t), x2(t), ..., xn(t)].
        V_vec: Vector field components [V^1(t), ..., V^n(t)] along the curve.
        t: Curve parameter symbol.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy Matrix of covariant derivative components dV^k/dt + \\Gamma^k_{ij} V^j (dx^i/dt).
    """
    dim = len(coords)
    gamma_vec = sp.Matrix(gamma_vec)
    V_vec = sp.Matrix(V_vec)
    chris = calculate_christoffel_symbols(g_matrix, coords)
    gamma_prime = [sp.diff(gamma_vec[i], t) for i in range(dim)]
    subs_map = {coords[i]: gamma_vec[i] for i in range(dim)}

    result = sp.zeros(dim, 1)
    for k in range(dim):
        term = sp.diff(V_vec[k], t)
        term += sum(
            V_vec[j] * gamma_prime[i] * chris[(i, j, k)].subs(subs_map)
            for i in range(dim)
            for j in range(dim)
        )
        result[k, 0] = sp.simplify(term)
    return result


def calculate_curve_acceleration(
    g_matrix: sp.Matrix,
    gamma_vec: sp.Matrix,
    t: sp.Symbol,
    coords: list,
) -> sp.Matrix:
    """
    Calculates the covariant acceleration of a parametrized curve gamma(t).

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        gamma_vec: Curve vector [x1(t), x2(t), ..., xn(t)].
        t: Curve parameter symbol.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy Matrix giving the covariant acceleration vector along gamma(t).
    """
    gamma_vec = sp.Matrix(gamma_vec)
    gamma_prime = sp.Matrix([sp.diff(gamma_vec[i], t) for i in range(len(coords))])
    return calculate_covariant_derivative_along_curve(g_matrix, gamma_vec, gamma_prime, t, coords)


def calculate_divergence(
    g_matrix: sp.Matrix,
    V_vec: sp.Matrix,
    coords: list,
) -> sp.Expr:
    """
    Calculates the divergence of a vector field with respect to a metric.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        V_vec: Vector field components [V^1(x), ..., V^n(x)].
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy expression for the divergence of V.
    """
    dim = len(coords)
    V_vec = sp.Matrix(V_vec)
    chris = calculate_christoffel_symbols(g_matrix, coords)

    divergence = sum(sp.diff(V_vec[i], coords[i]) for i in range(dim))
    divergence += sum(
        chris[(i, j, i)] * V_vec[j]
        for i in range(dim)
        for j in range(dim)
    )
    return sp.simplify(divergence)


def calculate_conductive_divergence(
    g_matrix: sp.Matrix,
    V_vec: sp.Matrix,
    W_matrix: sp.Matrix,
    coords: list,
) -> sp.Expr:
    """
    Calculates the divergence of the conductivity-weighted vector field W*V.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        V_vec: Vector field components [V^1(x), ..., V^n(x)].
        W_matrix: Conductivity matrix W(x) in coordinate symbols.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy expression for the W-divergence of V.
    """
    V_vec = sp.Matrix(V_vec)
    U_vec = W_matrix * V_vec
    return calculate_divergence(g_matrix, U_vec, coords)


def calculate_gradient(
    g_matrix: sp.Matrix,
    f: sp.Expr,
    coords: list,
) -> sp.Matrix:
    """
    Calculates the gradient of a scalar function f with respect to metric g.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        f: Scalar function f(x1, ..., xn).
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy Matrix of gradient components [g^{k el} \\partial_el f].
    """
    g_inv = g_matrix.inv()
    grad = sp.zeros(len(coords), 1)
    for k in range(len(coords)):
        grad[k, 0] = sum(
            g_inv[k, el] * sp.diff(f, coords[el])
            for el in range(len(coords))
        )
    return grad


def calculate_hessian(
    g_matrix: sp.Matrix,
    f: sp.Expr,
    coords: list,
) -> sp.Matrix:
    """
    Calculates the Hessian of a scalar function f with respect to metric g.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        f: Scalar function f(x1, ..., xn).
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy Matrix with components H_{ij} = f_{,ij} - Gamma^k_{ij} f_{,k}.
    """
    dim = len(coords)
    chris = calculate_christoffel_symbols(g_matrix, coords)
    H = sp.zeros(dim, dim)
    f_grad = [sp.diff(f, coords[k]) for k in range(dim)]
    for i in range(dim):
        for j in range(dim):
            H[i, j] = sp.diff(f_grad[j], coords[i])
            H[i, j] -= sum(chris[(i, j, k)] * f_grad[k] for k in range(dim))
            H[i, j] = sp.simplify(H[i, j])
    return H


def calculate_conductive_hessian(
    g_matrix: sp.Matrix,
    f: sp.Expr,
    W_matrix: sp.Matrix,
    coords: list,
) -> sp.Matrix:
    """
    Calculates the conductive Hessian of f using conductivity matrix W.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        f: Scalar function f(x1, ..., xn).
        W_matrix: Conductivity matrix W(x) in coordinate symbols.
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy Matrix for the W-Hessian: W * Hessian(f).
    """
    H = calculate_hessian(g_matrix, f, coords)
    return sp.simplify(W_matrix * H)


def calculate_laplacian(
    g_matrix: sp.Matrix,
    f: sp.Expr,
    coords: list,
) -> sp.Expr:
    """
    Calculates the Laplacian of a scalar function f with respect to metric g.

    Args:
        g_matrix: Metric tensor matrix in coordinate symbols.
        f: Scalar function f(x1, ..., xn).
        coords: Coordinate symbol list [x1, x2, ..., xn].

    Returns:
        A sympy expression for the Laplacian of f.
    """
    grad_f = calculate_gradient(g_matrix, f, coords)
    return calculate_divergence(g_matrix, grad_f, coords)

