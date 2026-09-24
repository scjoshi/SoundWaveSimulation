# Imaging Methods: 1D Ultrasound Simulations

This package contains two one-dimensional acoustic-wave simulations and the
figures, receiver traces, and conversation transcript produced while developing
them. The simulations are educational models, not validated clinical tools.

## Directory layout

```text
Imaging Methods/
|-- README.md
|-- scripts/
|   |-- 1DWaveSimulation.py
|   |-- 1DWaveSimulationDensity.py
|   |-- pyproject.toml
|   |-- uv.lock
|   `-- utilities/
|       `-- build_chat_transcript.py
`-- outputs/
    |-- data/
    |-- documents/
    `-- figures/
```

## Requirements

- Windows, macOS, or Linux
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

`uv` will obtain a compatible Python 3.12 interpreter and install NumPy,
Matplotlib, and the locked dependencies. Do not copy another person's virtual
environment; create a fresh one on the destination computer.

### Install uv

Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Alternatively, use Astral's official Windows installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal after installation and verify it:

```text
uv --version
```

## Set up the project

Extract the shared archive, open a terminal in the extracted `Imaging Methods`
directory, and enter the scripts directory:

```powershell
cd scripts
uv sync
```

The first sync creates `scripts/.venv` and installs the exact dependency set
recorded in `uv.lock`.

## Run the original wave demonstration

Plot four snapshots of a wave crossing a sound-speed interface:

```powershell
uv run python .\1DWaveSimulation.py
```

Animate the original simulation:

```powershell
uv run python .\1DWaveSimulation.py --animate
```

Change the animation frame delay:

```powershell
uv run python .\1DWaveSimulation.py --animate --interval 40
```

## Run the virtual ultrasound experiment

This is the main demonstration. It performs water-only calibration,
water-skin-water through-transmission, and pulse-echo measurements. It then
estimates skin sound speed, pressure reflection coefficient, acoustic impedance,
and density.

Open the interactive measurement dashboard:

```powershell
uv run python .\1DWaveSimulationDensity.py
```

Run the numerical experiment without opening a window:

```powershell
uv run python .\1DWaveSimulationDensity.py --no-show
```

Add reproducible Gaussian receiver noise at 0.2% of each trace's peak:

```powershell
uv run python .\1DWaveSimulationDensity.py --noise 0.002
```

Generate a new dashboard and CSV file in the shared outputs directories:

```powershell
uv run python .\1DWaveSimulationDensity.py `
  --noise 0.002 `
  --save-csv ..\outputs\data\my_measurement_traces.csv `
  --save-figure ..\outputs\figures\my_measurement_dashboard.png `
  --no-show
```

On macOS or Linux, replace PowerShell backticks with backslashes, and use `/`
instead of `\` in paths.

## Included results

- `outputs/figures/ultrasound_measurement_demo.png`: example dashboard
- `outputs/data/ultrasound_measurement_traces.csv`: example receiver A-scans
- `outputs/documents/imaging_methods_chat_transcript.pdf`: development transcript

The shorter `dashboard.png` and `traces.csv` filenames are a noisy example
generated with `--noise 0.002`.

## Model assumptions

The enhanced simulation treats skin as a homogeneous, dermis-like, lossless
layer at normal incidence. Real skin is multilayered, attenuating, dispersive,
and anisotropic. The output should therefore be used to explain measurement
principles rather than predict clinical measurements directly.
