"""Virtual 1D ultrasound measurements of a water-coupled skin sample.

Runs through-transmission (water reference versus skin sample) and pulse-echo
experiments using staggered-grid acoustic pressure/particle-velocity FDTD.
The model is educational, lossless, normal-incidence, and one-dimensional.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Geometry and grid (SI units).
NX = 1000
DOMAIN_LENGTH_M = 0.040
DX = DOMAIN_LENGTH_M / NX
SKIN_START_M = 0.015
SKIN_END_M = 0.020
SKIN_THICKNESS_M = SKIN_END_M - SKIN_START_M
SOURCE_X_M = 0.005
ECHO_RECEIVER_X_M = 0.007
TRANSMISSION_RECEIVER_X_M = 0.030

# Numerics and source. At 2 MHz, water has ~19 grid cells per wavelength.
CFL = 0.90
NT = 1400
CENTER_FREQUENCY_HZ = 2.0e6
PULSE_CYCLES = 3
SOURCE_AMPLITUDE_PA = 1.0
SPONGE_CELLS = 60

# Water: NIST/IAPWS at 25 C and 0.1 MPa. Skin: dermis-like approximation.
WATER_DENSITY = 997.047
WATER_SOUND_SPEED = 1496.699
SKIN_DENSITY = 1109.0
SKIN_SOUND_SPEED = 1595.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Virtual ultrasound measurements of water-coupled skin"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, metavar="FRACTION",
        help="Gaussian receiver noise as a fraction of each trace peak",
    )
    parser.add_argument(
        "--save-csv", type=Path, metavar="PATH",
        help="Export time and receiver traces to CSV",
    )
    parser.add_argument(
        "--save-figure", type=Path, metavar="PATH",
        help="Save the measurement dashboard as an image",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Run without opening the dashboard window",
    )
    return parser.parse_args()


def cell_index(position_m):
    return int(np.clip(position_m / DX, 0, NX - 1))


def material_arrays(include_skin):
    """Return cell properties and harmonic-mean face density."""
    x = (np.arange(NX) + 0.5) * DX
    skin = ((x >= SKIN_START_M) & (x < SKIN_END_M)
            if include_skin else np.zeros(NX, dtype=bool))
    density = np.where(skin, SKIN_DENSITY, WATER_DENSITY)
    sound_speed = np.where(skin, SKIN_SOUND_SPEED, WATER_SOUND_SPEED)
    bulk_modulus = density * sound_speed**2
    impedance = density * sound_speed
    density_face = 2.0 * density[:-1] * density[1:] / (
        density[:-1] + density[1:]
    )
    return x, density, sound_speed, bulk_modulus, impedance, density_face


def build_sponge():
    """Smooth damping masks that suppress artificial boundary echoes."""
    damping_p = np.ones(NX)
    distance = np.linspace(1.0, 0.0, SPONGE_CELLS, endpoint=False)
    edge = np.exp(-0.10 * distance**2)
    damping_p[:SPONGE_CELLS] *= edge
    damping_p[-SPONGE_CELLS:] *= edge[::-1]
    return damping_p, np.sqrt(damping_p[:-1] * damping_p[1:])


def source_waveform(time_s):
    """Three-cycle Gaussian-windowed pressure burst."""
    duration = PULSE_CYCLES / CENTER_FREQUENCY_HZ
    center = 2.5 * duration
    sigma = duration / 3.0
    envelope = np.exp(-0.5 * ((time_s - center) / sigma) ** 2)
    carrier = np.sin(2.0 * np.pi * CENTER_FREQUENCY_HZ * (time_s - center))
    return SOURCE_AMPLITUDE_PA * carrier * envelope


def advance_wave(pressure, velocity, bulk_modulus, density_face,
                 damping_p, damping_v, dt):
    """Advance pressure and face-centered particle velocity one step."""
    velocity -= (dt / (density_face * DX)) * np.diff(pressure)
    velocity_padded = np.pad(velocity, (1, 1), mode="edge")
    pressure -= (bulk_modulus * dt / DX) * np.diff(velocity_padded)
    pressure *= damping_p
    velocity *= damping_v


def simulate(include_skin):
    """Fire the source once and record pulse-echo and transmission A-scans."""
    properties = material_arrays(include_skin)
    x, density, sound_speed, bulk_modulus, impedance, density_face = properties
    # Calibration and sample acquisitions must share one sampling clock. Use
    # the fastest material in either experiment to satisfy CFL in both.
    dt = CFL * DX / max(WATER_SOUND_SPEED, SKIN_SOUND_SPEED)
    time_s = np.arange(NT) * dt
    source = source_waveform(time_s)
    damping_p, damping_v = build_sponge()
    pressure = np.zeros(NX)
    velocity = np.zeros(NX - 1)
    source_i = cell_index(SOURCE_X_M)
    echo_i = cell_index(ECHO_RECEIVER_X_M)
    transmission_i = cell_index(TRANSMISSION_RECEIVER_X_M)
    echo_trace = np.empty(NT)
    transmission_trace = np.empty(NT)

    for step in range(NT):
        advance_wave(pressure, velocity, bulk_modulus, density_face,
                     damping_p, damping_v, dt)
        pressure[source_i] += source[step]  # soft pressure source
        echo_trace[step] = pressure[echo_i]
        transmission_trace[step] = pressure[transmission_i]

    return {
        "x": x, "density": density, "sound_speed": sound_speed,
        "impedance": impedance, "dt": dt, "time": time_s, "source": source,
        "echo": echo_trace, "transmission": transmission_trace,
    }


def analytic_envelope(signal):
    """Hilbert-transform envelope implemented with NumPy's FFT."""
    spectrum = np.fft.fft(signal)
    weights = np.zeros(signal.size)
    weights[0] = 1.0
    if signal.size % 2 == 0:
        weights[1:signal.size // 2] = 2.0
        weights[signal.size // 2] = 1.0
    else:
        weights[1:(signal.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * weights))


def window_indices(time_s, center_s, half_width_s):
    indices = np.flatnonzero(
        (time_s >= center_s - half_width_s)
        & (time_s <= center_s + half_width_s)
    )
    if not indices.size:
        raise ValueError("Measurement gate is outside the simulated time range")
    return indices


def peak_in_window(trace, time_s, center_s, half_width_s):
    indices = window_indices(time_s, center_s, half_width_s)
    envelope = analytic_envelope(trace[indices])
    peak_index = indices[np.argmax(envelope)]
    return time_s[peak_index], envelope.max(), indices


def relative_amplitude(reference, target):
    """Signed target/reference amplitude from waveform correlation."""
    reference = reference - np.mean(reference)
    target = target - np.mean(target)
    correlation = np.correlate(target, reference, mode="full")
    peak = correlation[np.argmax(np.abs(correlation))]
    return peak / np.dot(reference, reference)


def cross_correlation_delay(reference, sample, dt):
    """Delay from pulse-envelope correlation, avoiding RF cycle ambiguity."""
    reference = analytic_envelope(reference)
    sample = analytic_envelope(sample)
    reference -= np.mean(reference)
    sample -= np.mean(sample)
    correlation = np.correlate(sample, reference, mode="full")
    lag_samples = np.argmax(np.abs(correlation)) - (reference.size - 1)
    return lag_samples * dt, lag_samples


def add_noise(results, fraction, rng):
    if fraction < 0:
        raise ValueError("--noise must be non-negative")
    for name in ("echo", "transmission"):
        if fraction:
            trace = results[name]
            sigma = fraction * np.max(np.abs(trace))
            results[name] = trace + rng.normal(0.0, sigma, trace.size)


def analyze(reference, sample):
    """Estimate c, Z, and density using only virtual receiver traces."""
    time_s, dt = sample["time"], sample["dt"]
    burst_center = 2.5 * PULSE_CYCLES / CENTER_FREQUENCY_HZ
    reference_arrival_center = burst_center + (
        TRANSMISSION_RECEIVER_X_M - SOURCE_X_M
    ) / WATER_SOUND_SPEED
    sample_arrival_center = burst_center + (
        SKIN_START_M - SOURCE_X_M
        + TRANSMISSION_RECEIVER_X_M - SKIN_END_M
    ) / WATER_SOUND_SPEED + SKIN_THICKNESS_M / SKIN_SOUND_SPEED
    reference_arrival, _, _ = peak_in_window(
        reference["transmission"], time_s, reference_arrival_center, 1.0e-6
    )
    sample_arrival, _, _ = peak_in_window(
        sample["transmission"], time_s, sample_arrival_center, 1.0e-6
    )
    delay_s = sample_arrival - reference_arrival
    lag_samples = round(delay_s / dt)
    c_transmission = SKIN_THICKNESS_M / (
        delay_s + SKIN_THICKNESS_M / WATER_SOUND_SPEED
    )

    incident_center = burst_center + (
        ECHO_RECEIVER_X_M - SOURCE_X_M
    ) / WATER_SOUND_SPEED
    front_center = burst_center + (
        SKIN_START_M - SOURCE_X_M + SKIN_START_M - ECHO_RECEIVER_X_M
    ) / WATER_SOUND_SPEED
    back_center = front_center + 2.0 * SKIN_THICKNESS_M / SKIN_SOUND_SPEED
    gate = 1.2e-6

    incident_time, incident_amp, incident_indices = peak_in_window(
        sample["echo"], time_s, incident_center, gate
    )
    front_time, front_amp, front_indices = peak_in_window(
        sample["echo"], time_s, front_center, gate
    )
    back_time, back_amp, back_indices = peak_in_window(
        sample["echo"], time_s, back_center, gate
    )
    c_echo = 2.0 * SKIN_THICKNESS_M / (back_time - front_time)

    incident = sample["echo"][incident_indices]
    front = sample["echo"][front_indices]
    common = min(incident.size, front.size)
    reflection = relative_amplitude(incident[:common], front[:common])
    water_z = WATER_DENSITY * WATER_SOUND_SPEED
    impedance = water_z * (1.0 + reflection) / (1.0 - reflection)
    density = impedance / c_echo
    true_z = SKIN_DENSITY * SKIN_SOUND_SPEED
    true_reflection = (true_z - water_z) / (true_z + water_z)

    return {
        "delay_s": delay_s, "lag_samples": lag_samples,
        "c_transmission": c_transmission, "c_echo": c_echo,
        "reflection": reflection, "impedance": impedance, "density": density,
        "true_impedance": true_z, "true_reflection": true_reflection,
        "incident_time": incident_time, "front_time": front_time,
        "back_time": back_time, "incident_amplitude": incident_amp,
        "front_amplitude": front_amp, "back_amplitude": back_amp,
    }


def percent_error(measured, true):
    return 100.0 * (measured - true) / true


def plot_dashboard(reference, sample, measured):
    time_us = sample["time"] * 1e6
    x_mm = sample["x"] * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x_mm, sample["sound_speed"], label="Sound speed (m/s)")
    ax.plot(x_mm, sample["density"], label="Density (kg/m³)")
    ax.axvspan(SKIN_START_M * 1e3, SKIN_END_M * 1e3,
               color="tan", alpha=0.25, label="Skin sample")
    ax.axvline(SOURCE_X_M * 1e3, color="black", linestyle=":", label="Source")
    ax.axvline(TRANSMISSION_RECEIVER_X_M * 1e3,
               color="purple", linestyle=":", label="Transmission receiver")
    ax.set(title="Virtual measurement geometry", xlabel="Position (mm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(time_us, reference["transmission"], label="Water reference")
    ax.plot(time_us, sample["transmission"], label="Skin sample", alpha=0.85)
    ax.set(
        title=f"Through-transmission delay: {measured['delay_s'] * 1e9:+.1f} ns",
        xlabel="Time (µs)", ylabel="Pressure (Pa)",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(time_us, sample["echo"], color="#1f77b4")
    for label, key, color in (
        ("Incident", "incident_time", "gray"),
        ("Front echo", "front_time", "tab:orange"),
        ("Back echo", "back_time", "tab:red"),
    ):
        ax.axvline(measured[key] * 1e6, color=color,
                   linestyle="--", label=label)
    ax.set(title="Pulse-echo A-scan", xlabel="Time (µs)", ylabel="Pressure (Pa)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "RECOVERED PROPERTIES", "",
        f"Sound speed (transmission) {measured['c_transmission']:8.1f} m/s  "
        f"error {percent_error(measured['c_transmission'], SKIN_SOUND_SPEED):+6.2f}%",
        f"Sound speed (pulse-echo)   {measured['c_echo']:8.1f} m/s  "
        f"error {percent_error(measured['c_echo'], SKIN_SOUND_SPEED):+6.2f}%",
        f"Pressure reflection        {measured['reflection']:+8.4f}      "
        f"true {measured['true_reflection']:+.4f}",
        f"Acoustic impedance         {measured['impedance'] / 1e6:8.3f} MRayl "
        f"error {percent_error(measured['impedance'], measured['true_impedance']):+6.2f}%",
        f"Density                    {measured['density']:8.1f} kg/m³ "
        f"error {percent_error(measured['density'], SKIN_DENSITY):+6.2f}%",
        "", "GROUND TRUTH",
        f"c = {SKIN_SOUND_SPEED:.1f} m/s",
        f"ρ = {SKIN_DENSITY:.1f} kg/m³",
        f"Z = {measured['true_impedance'] / 1e6:.3f} MRayl",
    ]
    ax.text(0.01, 0.98, "\n".join(lines), va="top",
            family="monospace", fontsize=10)
    fig.suptitle("Virtual ultrasound characterization of human skin", fontsize=15)
    return fig


def export_csv(path, reference, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "time_s", "source_pressure_pa", "water_reference_transmission_pa",
            "skin_sample_transmission_pa", "skin_sample_pulse_echo_pa",
        ])
        writer.writerows(zip(
            sample["time"], sample["source"], reference["transmission"],
            sample["transmission"], sample["echo"],
        ))


def print_results(measured):
    print("Virtual ultrasound measurement results")
    print(f"  Through-transmission delay:       {measured['delay_s'] * 1e9:+.2f} ns")
    print(f"  Skin sound speed (transmission):  {measured['c_transmission']:.2f} m/s")
    print(f"  Skin sound speed (pulse-echo):    {measured['c_echo']:.2f} m/s")
    print(f"  Pressure reflection coefficient: {measured['reflection']:+.5f}")
    print(f"  Skin acoustic impedance:          {measured['impedance'] / 1e6:.5f} MRayl")
    print(f"  Skin density:                     {measured['density']:.2f} kg/m^3")


def main():
    args = parse_args()
    reference = simulate(include_skin=False)
    sample = simulate(include_skin=True)
    rng = np.random.default_rng(20260902)
    add_noise(reference, args.noise, rng)
    add_noise(sample, args.noise, rng)
    measured = analyze(reference, sample)
    print_results(measured)

    if args.save_csv:
        export_csv(args.save_csv, reference, sample)
    if not args.no_show or args.save_figure:
        figure = plot_dashboard(reference, sample, measured)
        if args.save_figure:
            args.save_figure.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.save_figure, dpi=180)
        if not args.no_show:
            plt.show()
        else:
            plt.close(figure)


if __name__ == "__main__":
    main()
