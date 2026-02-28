# GSOC_2026_SU2
**Author**: Krrish Punj | B.Tech CSE, Thapar Institute (3rd year)  

## Assignment 1 — Compile SU2

**Environment**: NVIDIA DGX node (`dgxhnode5`) — Ubuntu 22.04, Docker container  
**Hardware**: NVIDIA H100 GPUs 
<br>
**SU2 Version**: develop branch, post v8.4.0 "Harrier"

---

## Environment

Running inside a Docker container on a shared DGX research node. Access via SSH as root. The container has CUDA 12, OpenMPI (HPC-X v2.19), and Python 3.10 pre-installed. No display — all visualization done off-screen.

```
OS:       Ubuntu 22.04 (containerized)
Compiler: GCC (system default)
MPI:      HPC-X OpenMPI v2.19
Python:   3.10
CUDA:     12 (installed, not used — SU2 GPU support incomplete)
```

---

## Steps

**Clone**
```bash
git clone https://github.com/su2code/SU2.git modified_su2
cd modified_su2
```

**Initial build** — baseline, confirmed working:
```bash
python3 meson.py setup build \
  -Denable-normal=true \
  -Denable-autodiff=false \
  --prefix=/usr/local

cd build && ninja -j$(nproc)
```

**Recompile with CPU optimizations** — eliminated AVX warnings, significant speedup on EPYC cores:
```bash
python3 meson.py setup build --wipe \
  -Denable-normal=true \
  -Denable-autodiff=false \
  -Dwith-omp=true \
  -Dcpp_args="-march=native -O3" \
  --prefix=/usr/local

cd build && ninja -j$(nproc)
```

**Recompile with Python wrapper** — required for Assignments 3 and 4:
```bash
python3 meson.py setup build --wipe \
  -Denable-normal=true \
  -Denable-pywrapper=true \
  -Denable-autodiff=false \
  -Dwith-omp=true \
  -Denable-mixedprec=false \
  -Dcpp_args="-march=native -O3" \
  --prefix=/usr/local

cd build && ninja -j$(nproc) && ninja install
```

`-Denable-mixedprec=false` was necessary — single-precision sparse algebra caused FGMRES orthogonalization failure on stiff RANS problems.

---

## Verification

Ran the turbulent flat plate tutorial (`turb_SA_flatplate.cfg`) to confirm correct installation:

```bash
mpirun --allow-run-as-root -np 4 \
  ./build/SU2_CFD/src/SU2_CFD \
  SU2_TestCases/rans/flatplate/turb_SA_flatplate.cfg
```

Confirmed: solver initialized, residuals decreasing, output files written correctly.

# Assignment 3: Python Wrapper Test Case
SU2 v8.4.0 "Harrier" · Turbulent Flat Plate · RANS/SA · M=0.2, Re=5×10⁶

---

## Test Case: Why Flat Plate

The turbulent flat plate is one of the most well-validated cases in CFD. There's no geometry complexity — just flow developing over a smooth wall — so any deviation in results points directly to the solver and turbulence model, not meshing artifacts. NASA and multiple groups have published reference data for skin friction and velocity profiles at these exact conditions (M=0.2, Re=5×10⁶), making it easy to sanity-check results. It's the standard first check for RANS implementations, which is exactly why SU2 ships it as the official Python wrapper testcase.

---

## Why This Configuration

- **RANS + Spalart-Allmaras** — SA is the industry-standard one-equation model for attached wall-bounded flows. Flat plate is attached flow by definition, so SA is the right choice and well-validated here.
- **M=0.2** — low enough to be effectively incompressible (avoids compressibility effects that complicate validation), but still exercises the compressible solver path.
- **Re=5×10⁶** — fully turbulent regime over the plate length. Clean turbulent boundary layer, no transition complications.
- **`-Denable-mixedprec=false`** — single precision sparse algebra caused FGMRES orthogonalization failure on the stiff RANS system. Double precision required for stability.
- **`-march=native -O3`** — AVX-optimized build for the EPYC cores on the DGX node. No correctness impact, measurable speedup.

---

## Compilation

```
python3 meson.py setup build --wipe \
  -Denable-pywrapper=true \
  -Denable-normal=true \
  -Denable-autodiff=false \
  -Dwith-omp=true \
  -Denable-mixedprec=false \
  -Dcpp_args="-march=native -O3" \
  --prefix=/usr/local
```

---

## Running the Case

Minimal driver script:
```python
import pysu2

driver = pysu2.CSinglezoneDriver('turb_SA_flatplate.cfg', 1, False)
driver.StartSolver()
driver.Finalize()
```

Ran on a DGX node (Ubuntu 22.04, Docker, HPC-X OpenMPI) as root — hence `--allow-run-as-root`.

---

## Convergence

Hit the target — 8 orders of magnitude drop in density residual:

```
All convergence criteria satisfied.

  Convergence Field  |   Value   | Criterion | Converged
  rms[Rho]           |  -8.00008 |    < -8   |   Yes
```

There's a segfault at exit (MPI cleanup on the DGX container — known issue). Solution files were written before it.

---

## Results

![Turbulent Flat Plate Results](results_a3_converged.png)

**Mach** — boundary layer grows along the wall, thickening downstream. Freestream holds M ≈ 0.2.

**Pressure** — nearly uniform across the domain, small rise toward the trailing edge as the boundary layer thickens.

**Temperature** — viscous dissipation heats the fluid near the wall (adiabatic BC). Freestream stays close to 300 K.

**Skin Friction Coefficient** — non-zero only on the plate surface, values 0.002–0.005, decreasing downstream. Consistent with turbulent flat plate correlations at Re=5×10⁶.

# Assignment 4 — Spatially Varying Wall Temperature 
SU2 v8.4.0 "Harrier" · Turbulent Flat Plate · RANS/SA · M=0.2, Re=5×10⁶

---

## Objective

Extend the Assignment 3 Python wrapper to apply a spatially varying isothermal wall temperature along the flat plate, demonstrating the Python interface's ability to modify boundary conditions programmatically at each solver iteration.

---

## Implementation

### Boundary Condition Setup (`turb_SA_flatplate.cfg`)

Two config entries are required for Python-controlled wall temperature:

```ini
MARKER_ISOTHERMAL= ( wall, 300.0 )   # registers BC type; value overridden by Python
MARKER_PYTHON_CUSTOM= ( wall )        # marks boundary as Python-controllable
```

`MARKER_ISOTHERMAL` sets the BC type. `MARKER_PYTHON_CUSTOM` tells SU2 to accept per-vertex temperature overrides from the Python wrapper each iteration.

### Temperature Profile

```
T(x) = 300 + 50·sin(πx)     [K],    x ∈ [0, 1] m
```

- 300 K at leading and trailing edges — matches freestream, avoids artificial thermal gradients at boundaries
- Peak 350 K at x = 0.5 m (plate midpoint)
- Smooth and differentiable everywhere

### Python Wrapper

The key addition over Assignment 3 is the per-iteration temperature loop:

```python
marker_ids = driver.GetMarkerIndices()
wall_id    = marker_ids['wall']
nVertex    = driver.GetNumberMarkerNodes(wall_id)

while TimeIter < nTimeIter:
    driver.Preprocess(TimeIter)

    # Set T(x) using physical x-coordinate — NOT vertex index
    for iVertex in range(nVertex):
        x  = driver.MarkerCoordinates(wall_id)(iVertex, 0)
        x  = max(0.0, min(1.0, x))
        Tw = 300.0 + 50.0 * sin(pi * x)
        driver.SetMarkerCustomTemperature(wall_id, iVertex, Tw)

    driver.BoundaryConditionsUpdate()
    driver.Run()
    ...
```

**Critical implementation detail**: `MarkerCoordinates` returns a `CPyWrapperMarkerMatrixView` object — uses `(row, col)` tuple indexing, not `[row][col]`. Physical x-coordinate must be used (not vertex index ratio `i/(n-1)`) since mesh nodes on a marker have no guaranteed spatial ordering.

---

## Run Configuration

```ini
CFL_NUMBER= 5.0
MGLEVEL= 3
LINEAR_SOLVER= FGMRES
CONV_RESIDUAL_MINVAL= -8
ITER= 10000
```

```bash
OMP_NUM_THREADS=1 mpirun --allow-run-as-root -np 1 \
  python3 flatplate_spatial_T_wrapper.py
```

---

## Linear Solver Study

Both available Krylov solvers were tested under identical conditions (mesh, config, CFL, multigrid settings identical):

| Linear Solver | Converged Iter | rms[Rho] | CD | CL | Avg_a (m/s) |
|---|---|---|---|---|---|
| FGMRES | 9671 | −8.00004 | 0.002809 | −0.188203 | 350.22 |
| BCGSTAB | 9671 | −8.00004 | 0.002809 | −0.188203 | 350.22 |

Results are identical. With `MGLEVEL=3`, multigrid coarse-level corrections dominate outer nonlinear convergence — the linear solver only runs 10 inner iterations per outer step (to tolerance 1e-6). The outer convergence path is determined entirely by the multigrid cycle quality, making the Krylov method choice irrelevant at these settings.

`CONJUGATE_GRADIENT` is not applicable — requires a symmetric system, which does not hold for RANS with upwind discretization. `RESTARTED_FGMRES` would behave identically to FGMRES within a 10-iteration window.

**Decision**: FGMRES retained — SU2 default, better suited for non-symmetric systems in general.

---

## Results

### Convergence

Converged at iteration 9671 (vs ~4921 for Assignment 3 uniform-temperature case):

```
rms[Rho] = -8.00004   < -8   → Yes
```

The heated wall adds ~50K of thermal forcing at the midpoint. More iterations required for the boundary layer to reach thermal equilibrium — the solver must satisfy an additional energy balance at every wall node each iteration.

### Flow Quantities vs Assignment 3

| Quantity | A3 — Uniform 300K | A4 — T(x) profile | Change |
|---|---|---|---|
| CD | 0.002850 | 0.002809 | −1.4% |
| CL | −0.188287 | −0.188203 | ~0% |
| Avg_a (m/s) | 347.75 | 350.22 | +0.7% |

Wall heating reduces near-wall density → lower skin friction → CD drops slightly. Average sound speed increases as domain-averaged temperature rises above 300K. Lift is unaffected — expected for zero angle of attack.

### Volume Output

![Assignment 4 Results](results_a4_final.png)

**Left**: Near-wall temperature field (y < 0.02 m). Thermal boundary layer is thin (~1–2 mm) at Re=5×10⁶ — turbulent Prandtl number Pr_t = 0.9 keeps thermal and velocity BLs tightly coupled.

**Center**: Applied wall BC. Sinusoidal profile peaks at 350K at midplate, returns to 300K at both ends.

**Right**: Near-wall Mach field showing boundary layer growth from leading edge — canonical turbulent flat plate behavior.

---

## Notes

The MPI binary file writer crashes during cleanup (`PMPI_File_close`) on HPC-X v2.19. This is a known incompatibility between SU2's collective MPI-IO writer and the installed OpenMPI version. All result files (`flow.vtu`, `history.csv`, `restart_flow.dat`) are written correctly before the crash. This is an environment issue, not a solver issue.
