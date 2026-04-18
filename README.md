# CEG8517 - State of the Art Modelling in Hydraulics: Coursework 2

## Description

A library containing components to create schemes for solving the Shallow Water Equations, created during the course of Coursework 2 for CEG8517 at Newcastle University.

This project implements an Object-Oriented structure with classes representing meshes, boundary conditions, physics, Riemann solvers, spatial reconstructors and temporal integrators. These classes are then used together within a simulation class, which has access to animation methods and other utilities to record the results of simulations.

## Example Usage

The project is configured to generate results for 4 test cases. Test case 3 is configured such that an animation will display whilst the solver is running, all test cases will output results in CSV format into an `exports` directory.

Detailed below are two methods of running the code in this repository. The code was developed using `uv` as the package and project manager, and I strongly recommend its use to avoid any unforeseen reproducibility issues.

### UV

If you do not have `uv` installed, you can find the full installation instructions in the [official Astral documentation](https://docs.astral.sh/uv/getting-started/installation/). The quickest methods to install it are:

**macOS/Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

Once `uv` is installed you can simply run the following commands in the repository directory:

```bash
uv sync
uv run main.py
```

### pip

If you don't want to install UV, you can still run the project uisng standard pip.

First, run:
```bash
python3 -m venv .venv
```

Then depending on your OS, run:

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.\.venv\Scripts\activate
```

Then:
```bash
pip install -r requirements.txt
```

Finally, run:
```bash
python main.py
```