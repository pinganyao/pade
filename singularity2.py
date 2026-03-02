import csv
import math

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


def prompt_singularity():
    while True:
        raw = input("Enter singularity x-value (e.g., 1, -pi/2): ").strip()
        try:
            singularity_expr = sp.sympify(raw)
            singularity_val = float(sp.N(singularity_expr))
        except (sp.SympifyError, TypeError, ValueError):
            print("Invalid singularity. Please enter a number or expression like pi/2.")
            continue
        if singularity_val == 0:
            print("Singularity cannot be zero.")
            continue
        return singularity_expr, singularity_val


def prompt_num_points():
    while True:
        raw = input("Enter number of sample points N (e.g., 1000): ").strip()
        try:
            num_points = int(raw)
        except ValueError:
            print("Please enter an integer value for N.")
            continue
        if num_points < 2:
            print("N must be at least 2.")
            continue
        return num_points


def build_log_spaced_points(singularity, num_points):
    radius = abs(singularity)
    base = 0.1 * radius
    delta = 0.000001 * radius
    ratio = delta / base
    xs = []
    for k in range(num_points):
        exponent = k / (num_points - 1)
        distance = base * (ratio ** exponent)
        if singularity > 0:
            x_val = singularity - distance
        else:
            x_val = singularity + distance
        xs.append(x_val)
    return xs


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


def write_log_growth_csv(filename, xs, distances, taylor_errors, pade_errors):
    eps = 1e-300
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["index", "x", "d", "err_taylor", "err_pade", "log_err_taylor", "log_err_pade"]
        )
        for idx, (x_val, d_val, err_t, err_p) in enumerate(
            zip(xs, distances, taylor_errors, pade_errors), start=1
        ):
            # Clamp errors before logging
            err_t_clamped = max(err_t, eps)
            err_p_clamped = max(err_p, eps)
            log_err_t = math.log(err_t_clamped)
            log_err_p = math.log(err_p_clamped)
            writer.writerow([
                idx,
                format_float(x_val),
                format_float(d_val),
                format_sigfigs(err_t, 4),
                format_sigfigs(err_p, 4),
                format_float(log_err_t),
                format_float(log_err_p)
            ])


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
    _, singularity_val = prompt_singularity()
    num_points = prompt_num_points()

    taylor_poly = sp.series(func_expr, x, 0, degree + 1).removeO()

    pade_order = degree // 2
    pade_expr = build_pade_from_taylor(taylor_poly, x, pade_order)

    try:
        original_func = sp.lambdify(x, func_expr, "math")
        taylor_func = sp.lambdify(x, taylor_poly, "math")
        pade_func = sp.lambdify(x, pade_expr, "math")
    except Exception as exc:
        raise RuntimeError("Failed to create numerical functions.") from exc

    xs = build_log_spaced_points(singularity_val, num_points)

    # Compute distances d_k = |singularity - x_k|
    distances = [abs(singularity_val - x_val) for x_val in xs]

    taylor_errors, taylor_max = compute_errors(original_func, taylor_func, xs)
    pade_errors, pade_max = compute_errors(original_func, pade_func, xs)

    taylor_csv = "taylor-singularity.csv"
    pade_csv = "pade-singularity.csv"

    write_csv(taylor_csv, xs, taylor_errors, taylor_max)
    write_csv(pade_csv, xs, pade_errors, pade_max)

    log_growth_csv = "log-growth.csv"
    write_log_growth_csv(log_growth_csv, xs, distances, taylor_errors, pade_errors)

    print("Finished.")
    print(f"Taylor MAE: {format_sigfigs(taylor_max)}")
    print(f"Pade MAE:   {format_sigfigs(pade_max)}")
    print(f"CSV files written: {taylor_csv}, {pade_csv}")
    print("Wrote log-growth.csv")


if __name__ == "__main__":
    main()

