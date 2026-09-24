"""Polak–Ribière nonlinear CG with an Armijo line search.

Search directions live on the interior mask. After each trial step the
medium is projected onto [c_min, c_max] and the known exterior. A negative
Polak–Ribière β restarts as steepest descent.
"""

import numpy as np


def masked_dot(a, b, mask):
    return float(np.dot(a[mask], b[mask]))


def polak_ribiere(
    m0,
    misfit,
    misfit_and_grad,
    project,
    mask,
    max_iter=15,
    armijo_c1=1e-4,
    shrink=0.5,
    max_line_search=12,
    step_fraction=0.05,
    verbose=True,
):
    """Minimize J(m) with Polak–Ribière CG.

    misfit(m) must return J; misfit_and_grad(m) returns (J, g).
    """
    m = project(np.asarray(m0, dtype=float))
    J, g = misfit_and_grad(m)
    history = {
        "misfit": [float(J)],
        "step_size": [0.0],
        "grad_norm": [np.sqrt(max(masked_dot(g, g, mask), 0.0))],
    }
    p = -g
    alpha_prev = None

    for it in range(max_iter):
        gnorm2 = masked_dot(g, g, mask)
        if gnorm2 <= 0.0:
            break
        mnorm = np.sqrt(max(masked_dot(m, m, mask), 0.0))

        accepted = False
        alpha = 0.0
        m_trial = m
        J_trial = J
        for restart in (False, True):
            if restart:
                p = -g
            gdotp = masked_dot(g, p, mask)
            if gdotp >= 0.0:
                p = -g
                gdotp = -gnorm2
            pnorm = np.sqrt(max(masked_dot(p, p, mask), 0.0))
            if pnorm <= 0.0:
                break
            if restart or alpha_prev is None:
                alpha = step_fraction * mnorm / pnorm
            else:
                alpha = alpha_prev * 1.5
            for _ in range(max_line_search):
                m_trial = project(m + alpha * p)
                J_trial = misfit(m_trial)
                if J_trial <= J + armijo_c1 * alpha * gdotp:
                    accepted = True
                    break
                alpha *= shrink
            if accepted:
                break

        if not accepted:
            if verbose:
                print("  CG line search failed; stopping.")
            break

        m = m_trial
        J_new, g_new = misfit_and_grad(m)
        denom = masked_dot(g, g, mask)
        beta = masked_dot(g_new, g_new - g, mask) / (denom + 1e-30)
        if beta < 0.0:
            beta = 0.0
        p = -g_new + beta * p
        g = g_new
        J = J_new
        alpha_prev = alpha
        gnorm = np.sqrt(max(masked_dot(g, g, mask), 0.0))
        history["misfit"].append(float(J))
        history["step_size"].append(float(alpha))
        history["grad_norm"].append(gnorm)
        if verbose:
            print(
                f"  CG iter {it + 1:3d}  J = {J:.6e}  "
                f"step = {alpha:.3e}  |g| = {gnorm:.4e}"
            )

    return m, history
