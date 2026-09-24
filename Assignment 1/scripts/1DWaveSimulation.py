"""
1D acoustic wave simulation using the Finite-Difference Time-Domain (FDTD) method.

Continuous model
----------------
We solve the 1D wave equation with a spatially varying sound speed c(x):

    d^2 u / dt^2 = c(x)^2 * d^2 u / dx^2

where u(x, t) is displacement / pressure, x is position, and t is time.

Discretization
--------------
The domain is sampled on a uniform spatial grid x_i = i * dx  (i = 0, ..., Nx-1)
and advanced in time with step dt. Three time levels are stored:

    u_old[i]  ~  u(x_i, t - dt)     (previous time step, n-1)
    u[i]      ~  u(x_i, t)          (current time step, n)
    u_next[i] ~  u(x_i, t + dt)     (next time step, n+1)

Second derivatives are replaced by second-order central finite differences:

    d^2 u / dt^2  ->  (u_next[i] - 2*u[i] + u_old[i]) / dt^2
    d^2 u / dx^2  ->  (u[i+1] - 2*u[i] + u[i-1]) / dx^2

Substituting into the wave equation and solving for u_next[i] gives the
leapfrog / explicit FDTD update used in advance_wave().

Stability requires the Courant-Friedrichs-Lewy (CFL) condition:

    c_max * dt / dx < 1

Domain edges use first-order Mur absorbing boundary conditions so outgoing
waves can leave the grid with minimal reflection.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Simulation Parameters (grid and time discretization) ---
# Spatial mesh: Nx points spaced by dx, covering physical length ~ (Nx - 1) * dx.
# Temporal mesh: Nt steps of size dt, covering physical time Nt * dt.
Nx = 800          # Number of spatial grid points (domain length ~ (Nx - 1) * dx)
Nt = 1000         # Number of time steps
dx = 0.5          # Spatial step size (distance between neighboring grid points)
dt = 0.25         # Time step size; must satisfy CFL: c_max * dt / dx < 1
INTERFACE_X = 200  # Medium boundary in physical coordinates


def parse_args():
    parser = argparse.ArgumentParser(description="1D sound wave simulation")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the wave propagation over time instead of plotting snapshots",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Delay between animation frames in milliseconds (default: 20)",
    )
    return parser.parse_args()


def advance_wave(u_old, u, u_next, c, dt, dx):
    # Interior points: explicit FDTD stencil for d^2u/dt^2 = c(x)^2 d^2u/dx^2
    #
    # Starting from the finite-difference approximations
    #   (u_next - 2*u + u_old) / dt^2 = c^2 * (u[i+1] - 2*u + u[i-1]) / dx^2
    # multiply by dt^2 and rearrange to obtain the update formula
    #   u_next[i] = 2*u[i] - u_old[i] + r^2 * (u[i+1] - 2*u[i] + u[i-1])
    # where r = c[i]*dt/dx is the local Courant number at grid point i.
    for i in range(1, Nx - 1):
        r2 = (c[i] * dt / dx) ** 2
        u_next[i] = 2.0 * u[i] - u_old[i] + r2 * (u[i + 1] - 2.0 * u[i] + u[i - 1])

    # First-order Mur absorbing boundary conditions at x=0 and x=Nx-1.
    # These discretize the one-way wave equation at each edge so energy can exit
    # the domain instead of reflecting. With alpha = (c*dt - dx)/(c*dt + dx):
    #   left:  u_next[0]  = u[1]  + alpha_left  * (u_next[1]  - u[0])
    #   right: u_next[-1] = u[-2] + alpha_right * (u_next[-2] - u[-1])
    alpha_left = (c[0] * dt - dx) / (c[0] * dt + dx)
    alpha_right = (c[-1] * dt - dx) / (c[-1] * dt + dx)
    u_next[0] = u[1] + alpha_left * (u_next[1] - u[0])
    u_next[-1] = u[-2] + alpha_right * (u_next[-2] - u[-1])


def setup_medium_and_initial_conditions():
    # --- 2. Define the Medium Interface ---
    # c(x) is stored on the same spatial grid as u. A jump in c at x = INTERFACE_X
    # enters the FDTD update through the local Courant number r = c[i]*dt/dx.
    # Left side (Medium 1): Fast speed (c = 1.0)
    # Right side (Medium 2): Slow speed (c = 0.5)
    c = np.ones(Nx) * 1.0
    c[Nx // 2:] = 0.5  # Boundary at x = INTERFACE_X

    # --- 3. Initialize Wave Fields ---
    u_old = np.zeros(Nx)   # Time step (n-1)
    u = np.zeros(Nx)       # Time step (n)
    u_next = np.zeros(Nx)  # Time step (n+1)

    # --- 4. Setup Initial Condition (Gaussian-windowed sine wave) ---
    # The second-order-in-time scheme needs u at two consecutive time levels:
    #   u_old  = u(x, 0)
    #   u      = u(x, dt)
    #
    # u_old is sampled directly from the continuous initial profile.
    # u is obtained by advecting that profile one time step to the right using
    # the local wave speed, which approximates du/dt for a right-traveling packet.
    x = np.arange(Nx) * dx
    x0 = 50.0        # Starting center of the sound pulse
    sigma = 5.0      # Spatial width of the Gaussian envelope
    wavelength = 10.0
    k = 2.0 * np.pi / wavelength  # Wavenumber of the carrier sine wave

    gaussian_t0 = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    gaussian_t1 = np.exp(-0.5 * ((x - x0 - c * dt) / sigma) ** 2)

    # u(x, 0) = sin(k*(x - x0)) * Gaussian(x; x0, sigma)
    u_old = np.sin(k * (x - x0)) * gaussian_t0
    # u(x, dt) = sin(k*(x - x0 - c*dt)) * Gaussian(x; x0 + c*dt, sigma)
    u = np.sin(k * (x - x0 - c * dt)) * gaussian_t1

    return x, c, u_old, u, u_next


def decorate_axes(ax):
    ax.axvline(
        x=INTERFACE_X,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Medium Boundary (x={INTERFACE_X})",
    )
    ax.text(
        INTERFACE_X / 4,
        0.8,
        "Medium 1\n(Fast: c=1.0)",
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    ax.text(
        INTERFACE_X + INTERFACE_X / 2,
        0.8,
        "Medium 2\n(Slow: c=0.5)",
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    ax.set_xlabel("Spatial Position (x)", fontsize=10)
    ax.set_ylabel("Wave Amplitude / Pressure (u)", fontsize=10)
    ax.set_xlim(0, (Nx - 1) * dx)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)


def run_snapshot_simulation(x, c, u_old, u, u_next):
    snapshots = {}
    track_steps = [0, 280, 520, 840]

    # Explicit time marching: at each step, compute u_next from (u_old, u),
    # then rotate the three time-level arrays forward by one step.
    for t in range(Nt):
        advance_wave(u_old, u, u_next, c, dt, dx)
        u_old, u, u_next = u, u_next, u_old

        if t in track_steps:
            snapshots[t] = u.copy()

    return snapshots, track_steps


def plot_snapshots(x, snapshots, track_steps):
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, step in enumerate(track_steps):
        plt.plot(x, snapshots[step], label=f"Time Step {step}", color=colors[i], lw=2)

    decorate_axes(plt.gca())
    plt.title("1D Sound Wave Simulation Across Changing Medium Velocity", fontsize=12)
    plt.legend(loc="lower left")
    plt.show()


def animate_simulation(x, c, u_old, u, u_next, interval):
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, u.copy(), color="#1f77b4", lw=2)
    decorate_axes(ax)
    ax.legend(loc="lower left")

    def update(frame):
        nonlocal u_old, u, u_next

        if frame > 0:
            advance_wave(u_old, u, u_next, c, dt, dx)
            u_old, u, u_next = u, u_next, u_old

        line.set_ydata(u)
        ax.set_title(f"1D Sound Wave Simulation (Time Step {frame})", fontsize=12)

    # Keep a reference to the animation object so it is not garbage-collected.
    anim = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=interval,
        repeat=True,
        blit=False,
    )
    fig._animation = anim
    plt.show()


def main():
    args = parse_args()
    x, c, u_old, u, u_next = setup_medium_and_initial_conditions()

    if args.animate:
        animate_simulation(x, c, u_old, u, u_next, args.interval)
    else:
        snapshots, track_steps = run_snapshot_simulation(x, c, u_old, u, u_next)
        plot_snapshots(x, snapshots, track_steps)


if __name__ == "__main__":
    main()
