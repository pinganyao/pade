import mpmath as mp
import sympy as sp


def format_float(value, decimals=10):
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def prompt_function(x):
    while True:
        raw = input("Enter original function in x: ").strip()
        try:
            expr = sp.sympify(raw, locals={"x": x})
        except (sp.SympifyError, TypeError, ValueError):
            print("Invalid expression. Please try again.")
            continue
        extra_symbols = expr.free_symbols - {x}
        if extra_symbols:
            print("Expression must only use variable x.")
            continue
        return expr


def prompt_degree():
    while True:
        raw = input("Enter degree of Taylor polynomial (6 or 10): ").strip()
        try:
            degree = int(raw)
        except ValueError:
            print("Please enter an integer (6 or 10).")
            continue
        if degree not in {6, 10}:
            print("Only degrees 6 and 10 are accepted.")
            continue
        return degree


def build_pade_from_taylor(taylor_poly, x, order):
    mp.mp.dps = 50
    numeric_poly = sp.N(taylor_poly, 50)
    poly = sp.Poly(sp.expand(numeric_poly), x)
    coeffs = [poly.nth(k) for k in range(0, 2 * order + 1)]
    mp_coeffs = [mp.mpf(str(sp.N(c, 50))) for c in coeffs]
    p_coeffs, q_coeffs = mp.pade(mp_coeffs, order, order)
    numerator = sum(sp.Float(p_coeffs[k], 30) * x**k for k in range(order + 1))
    denominator = sum(sp.Float(q_coeffs[k], 30) * x**k for k in range(order + 1))
    return numerator, denominator, numerator / denominator


def find_real_roots(denominator, x, tol=1e-8):
    poly = sp.Poly(sp.expand(denominator), x)
    if poly.degree() <= 0:
        return []
    roots = []
    for root in poly.nroots(n=50):
        if abs(sp.im(root)) <= tol:
            roots.append(float(sp.re(root)))
    roots.sort()
    unique = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1e-6:
            unique.append(root)
    return unique


def main():
    x = sp.symbols("x")

    func_expr = prompt_function(x)
    degree = prompt_degree()

    taylor_poly = sp.series(func_expr, x, 0, degree + 1).removeO()
    pade_order = degree // 2
    _, denominator, _ = build_pade_from_taylor(taylor_poly, x, pade_order)

    roots = find_real_roots(denominator, x)

    if not roots:
        print("No Pole")
        return

    for root in roots:
        print(format_float(root))


if __name__ == "__main__":
    main()
