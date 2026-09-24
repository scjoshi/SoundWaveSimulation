"""
1D acoustic wave simulation using the Finite-Difference Time-Domain (FDTD) method.

Continuous model
----------------
We solve the 1D acoustic wave equation for pressure u(x, t) in a medium with
spatially varying sound speed c(x) and density rho(x):

    d^2 u / dt^2 = rho(x) * c(x)^2 * d/dx ( (1 / rho(x)) * du/dx )

The bulk modulus is K = rho c^2. Reflection and transmission at an interface
are controlled by the acoustic impedance Z = rho * c, not by c alone:

    R = (Z2 - Z1) / (Z2 + Z1)
    T = 2 Z2 / (Z2 + Z1)

If density is constant, the equation reduces to the familiar form
d^2 u / dt^2 = c(x)^2 d^2 u / dx^2.

Discretization
--------------
The domain is sampled on a uniform spatial grid x_i = i * dx  (i = 0, ..., Nx-1)
and advanced in time with step dt. Three time levels are stored:

    u_old[i]  ~  u(x_i, t - dt)     (previous time step, n-1)
    u[i]      ~  u(x_i, t)          (current time step, n)
    u_next[i] ~  u(x_i, t + dt)     (next time step, n+1)

Time is discretized with a second-order central difference:

    d^2 u / dt^2  ->  (u_next[i] - 2*u[i] + u_old[i]) / dt^2

The spatial operator uses a conservative staggered difference so that jumps
in density are handled at half-points x_{i+1/2}:

    rho_{i+1/2} = (rho[i] + rho[i+1]) / 2
    flux_{i+1/2} = (1 / rho_{i+1/2}) * (u[i+1] - u[i]) / dx
    d/dx (flux)_i = (flux_{i+1/2} - flux_{i-1/2}) / dx

Then:

    u_next[i] = 2*u[i] - u_old[i] + dt^2 * rho[i] * c[i]^2 * d/dx(flux)_i

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
Nx = 800           # Number of spatial grid points
Nt = 5000          # Number of time steps (scaled with dt to keep the same physical time)
dx = 0.5           # Spatial step size
dt = 0.05          # Time step size; must satisfy CFL: c_max * dt / dx < 1
INTERFACE_X = 200   # First medium boundary in physical coordinates
INTERFACE_X2 = 300  # Second medium boundary in physical coordinates

C_FAST = 1.0
C_SLOW = 0.75
C_VERY_FAST = 2.5

# Densities. Impedance Z = rho * c sets interface reflection/transmission.
RHO_SLOW = 1.0
RHO_FAST = 2.0
RHO_VERY_FAST = 0.4

PULSE_X0 = 50.0
PULSE_SIGMA = 5.0
PULSE_WAVELENGTH = 10.0
SNAPSHOT_STEPS = (0, 1400, 2600, 4200)
ANIMATION_STRIDE = 5  # Physics steps per animation frame (matches the dt reduction)
A_MODE_STEPS = 13000  # Long enough to receive echoes from both material interfaces
A_MODE_SENSOR_X = PULSE_X0


def parse_args():
    parser = argparse.ArgumentParser(description="1D sound wave simulation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--animate",
        action="store_true",
        help="Animate the wave propagation over time instead of plotting snapshots",
    )
    mode.add_argument(
        "--amode",
        action="store_true",
        help="Plot a pulse-echo A-mode response recorded near the source",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=20,
        help="Delay between animation frames in milliseconds (default: 20)",
    )
    return parser.parse_args()


class WaveSimulator:
    """Vectorized 1D FDTD solver with density and first-order Mur ABCs."""

    def __init__(self, c, rho, u_old, u, dx, dt):
        self.c = c
        self.rho = rho
        self.dx = dx
        self.dt = dt

        # dt^2 * K(x), with bulk modulus K = rho c^2
        self.stiffness = (dt ** 2) * rho * c ** 2
        # 1 / rho at half-points i+1/2, using arithmetic averaging
        self.inv_rho_half = 2.0 / (rho[1:] + rho[:-1])

        self.alpha_left = (c[0] * dt - dx) / (c[0] * dt + dx)
        self.alpha_right = (c[-1] * dt - dx) / (c[-1] * dt + dx)

        self.u_old = u_old
        self.u = u
        self.u_next = np.empty_like(u)

    def step(self):
        """Advance one time step using the variable-density FDTD stencil."""
        u = self.u
        u_old = self.u_old
        u_next = self.u_next
        dx = self.dx

        # Conservative spatial operator: d/dx ( (1/rho) du/dx )
        flux = self.inv_rho_half * (u[1:] - u[:-1]) / dx
        divergence = (flux[1:] - flux[:-1]) / dx

        # Interior points: u_tt = rho c^2 * d/dx((1/rho) u_x)
        u_next[1:-1] = 2.0 * u[1:-1] - u_old[1:-1] + self.stiffness[1:-1] * divergence

        # First-order Mur ABCs at x=0 and x=Nx-1 (use local sound speed).
        u_next[0] = u[1] + self.alpha_left * (u_next[1] - u[0])
        u_next[-1] = u[-2] + self.alpha_right * (u_next[-2] - u[-1])

        self.u_old, self.u, self.u_next = u, u_next, u_old
        return self.u


def piecewise_medium(x, values):
    """Assign a property that is constant in each of the three media."""
    left, middle, right = values
    return np.select(
        [x < INTERFACE_X, x < INTERFACE_X2],
        [left, middle],
        default=right,
    )


def setup_medium_and_initial_conditions():
    # --- 2. Define the three media ---
    # Each medium has a sound speed c and density rho. The FDTD update uses both
    # through K = rho c^2 and the (1/rho) spatial operator. Impedance Z = rho c
    # determines how much of the pulse reflects vs transmits at each interface.
    #   x < 200:        slow,      c = C_SLOW,      rho = RHO_SLOW
    #   200 <= x < 300: fast,      c = C_FAST,      rho = RHO_FAST
    #   x >= 300:       very fast, c = C_VERY_FAST, rho = RHO_VERY_FAST
    x = np.arange(Nx) * dx
    c = piecewise_medium(x, (C_SLOW, C_FAST, C_VERY_FAST))
    rho = piecewise_medium(x, (RHO_SLOW, RHO_FAST, RHO_VERY_FAST))

    # --- 3. Setup Initial Condition (Gaussian-windowed sine wave) ---
    # The second-order-in-time scheme needs u at two consecutive time levels:
    #   u_old = u(x, 0)
    #   u     = u(x, dt)
    #
    # u_old is sampled from the continuous initial profile. u is obtained by
    # advecting that profile one time step to the right, approximating a
    # right-traveling packet.
    k = 2.0 * np.pi / PULSE_WAVELENGTH
    gaussian_t0 = np.exp(-0.5 * ((x - PULSE_X0) / PULSE_SIGMA) ** 2)
    gaussian_t1 = np.exp(-0.5 * ((x - PULSE_X0 - c * dt) / PULSE_SIGMA) ** 2)

    u_old = np.sin(k * (x - PULSE_X0)) * gaussian_t0
    u = np.sin(k * (x - PULSE_X0 - c * dt)) * gaussian_t1
    return x, c, rho, u_old, u


def medium_label(name, c, rho):
    impedance = rho * c
    return f"{name}\n(c={c:g}, ρ={rho:g}, Z={impedance:g})"


def decorate_axes(ax):
    for boundary in (INTERFACE_X, INTERFACE_X2):
        ax.axvline(
            x=boundary,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Medium Boundary (x={boundary})",
        )
    ax.text(
        INTERFACE_X / 2,
        0.8,
        medium_label("Medium 1 (slow)", C_SLOW, RHO_SLOW),
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    ax.text(
        (INTERFACE_X + INTERFACE_X2) / 2,
        0.8,
        medium_label("Medium 2 (fast)", C_FAST, RHO_FAST),
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    ax.text(
        (INTERFACE_X2 + (Nx - 1) * dx) / 2,
        0.8,
        medium_label("Medium 3 (very fast)", C_VERY_FAST, RHO_VERY_FAST),
        horizontalalignment="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    ax.set_xlabel("Spatial Position (x)", fontsize=10)
    ax.set_ylabel("Wave Amplitude / Pressure (u)", fontsize=10)
    ax.set_xlim(0, (Nx - 1) * dx)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)


def run_snapshot_simulation(sim):
    snapshots = {}
    track_steps = set(SNAPSHOT_STEPS)

    for t in range(Nt):
        field = sim.step()
        if t in track_steps:
            snapshots[t] = field.copy()

    return snapshots, SNAPSHOT_STEPS


def plot_snapshots(x, snapshots, track_steps):
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, step in enumerate(track_steps):
        plt.plot(x, snapshots[step], label=f"Time Step {step}", color=colors[i], lw=2)

    decorate_axes(plt.gca())
    plt.title("1D Sound Wave Simulation with Variable Speed and Density", fontsize=12)
    plt.legend(loc="lower left")
    plt.show()


def animate_simulation(x, sim, interval):
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, sim.u, color="#1f77b4", lw=2)
    decorate_axes(ax)
    ax.legend(loc="lower left")

    def update(frame):
        if frame > 0:
            for _ in range(ANIMATION_STRIDE):
                sim.step()
        time_step = frame * ANIMATION_STRIDE
        line.set_ydata(sim.u)
        ax.set_title(f"1D Sound Wave Simulation (Time Step {time_step})", fontsize=12)

    anim = FuncAnimation(
        fig,
        update,
        frames=Nt // ANIMATION_STRIDE,
        interval=interval,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    fig._animation = anim
    plt.show()


def signal_envelope(signal):
    """Return the magnitude of the analytic signal using an FFT Hilbert transform."""
    sample_count = len(signal)
    spectrum = np.fft.fft(signal)
    hilbert_filter = np.zeros(sample_count)
    hilbert_filter[0] = 1.0

    if sample_count % 2 == 0:
        hilbert_filter[1 : sample_count // 2] = 2.0
        hilbert_filter[sample_count // 2] = 1.0
    else:
        hilbert_filter[1 : (sample_count + 1) // 2] = 2.0

    return np.abs(np.fft.ifft(spectrum * hilbert_filter))


def round_trip_time_to_depth(elapsed_time, sensor_x):
    """Convert echo return time to depth using the known layered sound speeds."""
    one_way_time = elapsed_time / 2.0
    first_layer_time = (INTERFACE_X - sensor_x) / C_SLOW
    second_layer_time = (INTERFACE_X2 - INTERFACE_X) / C_FAST

    depth = np.empty_like(one_way_time)
    in_first_layer = one_way_time <= first_layer_time
    in_second_layer = (
        (one_way_time > first_layer_time)
        & (one_way_time <= first_layer_time + second_layer_time)
    )
    in_third_layer = ~(in_first_layer | in_second_layer)

    depth[in_first_layer] = sensor_x + C_SLOW * one_way_time[in_first_layer]
    depth[in_second_layer] = (
        INTERFACE_X
        + C_FAST * (one_way_time[in_second_layer] - first_layer_time)
    )
    depth[in_third_layer] = (
        INTERFACE_X2
        + C_VERY_FAST
        * (
            one_way_time[in_third_layer]
            - first_layer_time
            - second_layer_time
        )
    )
    return depth


def run_amode_simulation(x, sim):
    """Record pressure at the source location to form a pulse-echo A-mode trace."""
    sensor_index = int(np.argmin(np.abs(x - A_MODE_SENSOR_X)))
    received_signal = np.empty(A_MODE_STEPS)

    # sim.u is the field at t=dt. Record it before each subsequent update so the
    # sample timestamps and solver state remain aligned.
    for sample in range(A_MODE_STEPS):
        received_signal[sample] = sim.u[sensor_index]
        sim.step()

    elapsed_time = (np.arange(A_MODE_STEPS) + 1) * dt
    depth = round_trip_time_to_depth(elapsed_time, x[sensor_index])
    return depth, received_signal, signal_envelope(received_signal), x[sensor_index]


def plot_amode_response(depth, received_signal, envelope, sensor_x):
    """Plot RF pressure and its detected envelope versus reconstructed depth."""
    scale = np.max(envelope)
    if scale > 0.0:
        received_signal = received_signal / scale
        envelope = envelope / scale

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        depth,
        received_signal,
        color="#7f7f7f",
        alpha=0.55,
        lw=0.8,
        label="RF pressure",
    )
    ax.plot(depth, envelope, color="#d62728", lw=2.0, label="Detected envelope")

    for boundary in (INTERFACE_X, INTERFACE_X2):
        ax.axvline(
            boundary,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Interface at x={boundary}" if boundary == INTERFACE_X else None,
        )

    ax.set_xlim(sensor_x, min(depth[-1], (Nx - 1) * dx))
    ax.set_xlabel("Reconstructed Depth (x)")
    ax.set_ylabel("Normalized Echo Amplitude")
    ax.set_title("A-mode Pulse-Echo Response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def main():
    args = parse_args()
    x, c, rho, u_old, u = setup_medium_and_initial_conditions()
    sim = WaveSimulator(c, rho, u_old, u, dx, dt)

    if args.animate:
        animate_simulation(x, sim, args.interval)
    elif args.amode:
        depth, received_signal, envelope, sensor_x = run_amode_simulation(x, sim)
        plot_amode_response(depth, received_signal, envelope, sensor_x)
    else:
        snapshots, track_steps = run_snapshot_simulation(sim)
        plot_snapshots(x, snapshots, track_steps)


if __name__ == "__main__":
    main()
