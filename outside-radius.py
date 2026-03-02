import csv
import math
import os

import mpmath as mp
import sympy as sp


def format_sigfigs(value, sig_figs=4):
    if value == 0 or value == 0.0:
        return "0"
    return f"{value:.{sig_figs}g}"


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


def prompt_radius():
    while True:
        raw = input("Enter radius of convergence (e.g., 1, pi/2): ").strip()
        try:
            radius_expr = sp.sympify(raw)
            radius_val = float(sp.N(radius_expr))
        except (sp.SympifyError, TypeError, ValueError):
            print("Invalid radius. Please enter a number or expression like pi/2.")
            continue
        if radius_val <= 0:
            print("Radius must be a positive value.")
            continue
        return radius_expr, radius_val


def prompt_step_size():
    while True:
        raw = input("Enter step size (decimal, e.g., 0.001): ").strip()
        try:
            step = float(raw)
        except ValueError:
            print("Invalid step size. Please enter a decimal number.")
            continue
        if step <= 0:
            print("Step size must be positive.")
            continue
        return step


def build_grid(start, end, step):
    values = []
    current = start
    guard = 0
    max_iters = int(math.ceil((end - start) / step)) + 3
    while current <= end + 1e-12 and guard <= max_iters:
        values.append(current)
        current += step
        guard += 1
    return values


def compute_errors(original_func, approx_func, xs):
    errors = []
    for x_val in xs:
        fx = original_func(x_val)
        ax = approx_func(x_val)
        err = abs(fx - ax)
        errors.append(err)
    max_error = max(errors) if errors else 0.0
    return errors, max_error


def write_csv(filename, xs, errors, max_error):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["index", "x-value", "pointwise error (4 s.f.)", "MAE (yes/no)"]
        )
        for idx, (x_val, err) in enumerate(zip(xs, errors), start=1):
            err_str = format_sigfigs(err, 4)
            is_max = "yes" if math.isclose(
                err, max_error, rel_tol=1e-12, abs_tol=1e-12
            ) else "no"
            writer.writerow([idx, format_float(x_val), err_str, is_max])


def unique_filename(filename):
    if not os.path.exists(filename):
        return filename
    base, ext = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = f"{base}-{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def build_pade_from_taylor(taylor_poly, x, order):
    mp.mp.dps = 50
    numeric_poly = sp.N(taylor_poly, 50)
    poly = sp.Poly(sp.expand(numeric_poly), x)
    coeffs = [poly.nth(k) for k in range(0, 2 * order + 1)]
    mp_coeffs = [mp.mpf(str(sp.N(c, 50))) for c in coeffs]
    p_coeffs, q_coeffs = mp.pade(mp_coeffs, order, order)
    numerator = sum(sp.Float(p_coeffs[k], 30) * x**k for k in range(order + 1))
    denominator = sum(sp.Float(q_coeffs[k], 30) * x**k for k in range(order + 1))
    return numerator / denominator


def main():
    x = sp.symbols("x")

    func_expr = prompt_function(x)
    degree = prompt_degree()
    _, radius_val = prompt_radius()
    step = prompt_step_size()

    taylor_poly = sp.series(func_expr, x, 0, degree + 1).removeO()

    pade_order = degree // 2
    pade_expr = build_pade_from_taylor(taylor_poly, x, pade_order)

    try:
        original_func = sp.lambdify(x, func_expr, "math")
        taylor_func = sp.lambdify(x, taylor_poly, "math")
        pade_func = sp.lambdify(x, pade_expr, "math")
    except Exception as exc:
        raise RuntimeError("Failed to create numerical functions.") from exc

    start = 1.2 * radius_val
    end = 2.0 * radius_val
    xs = build_grid(start, end, step)

    taylor_errors, taylor_max = compute_errors(original_func, taylor_func, xs)
    pade_errors, pade_max = compute_errors(original_func, pade_func, xs)

    taylor_csv = unique_filename("taylor-outside-radius.csv")
    pade_csv = unique_filename("pade-outside-radius.csv")

    write_csv(taylor_csv, xs, taylor_errors, taylor_max)
    write_csv(pade_csv, xs, pade_errors, pade_max)

    print("Finished.")
    print(f"Taylor MAE: {format_sigfigs(taylor_max)}")
    print(f"Pade MAE:   {format_sigfigs(pade_max)}")
    print(f"CSV files written: {taylor_csv}, {pade_csv}")


if __name__ == "__main__":
    main()
