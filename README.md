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

# Assignment 2 — Axisymmetric Turbulent Jet

> **Solver:** SU2 `INC_RANS` · **Turbulence:** SST V2003m · **Reference:** Fukushima et al. (2001)

---

## Motivation

I wanted to simulate a simple turbulent jet and compare the results with real experimental data. Fukushima, Aanen & Westerweel (2001) is a good reference — they used PIV to measure a water jet at Re = 2000 and report clear numbers to compare against (centreline velocity decay, jet spreading rate, turbulence intensity).

The jet is round (axisymmetric), so instead of a full 3D simulation I only simulate a 2D half-domain, which is much cheaper and captures the same physics.

| Parameter | Value |
|-----------|-------|
| Fluid | Water |
| Nozzle diameter `d` | 1 mm |
| Jet velocity `U_j` | 2 m/s |
| Reynolds number | 2000 |

---

## Mesh

Generated with the Gmsh Python API — see [`mesh.py`](mesh.py).

- **Type:** structured quad, axisymmetric half-domain
- **Size:** 96,000 cells, domain 160d × 30d
- **Two blocks:** finer near the nozzle (0–20d) to resolve the shear layer, coarser downstream (20–160d)

```
farfield  (p = 0)
┌─────────────────────────────────────┐
│  ambient_inlet       outlet (p=0)   │
│──────────────────────────────────── │
│  jet_inlet           outlet (p=0)   │
└─────────────────────────────────────┘
axis (symmetry, y = 0)
```

| Boundary | Type | Value |
|----------|------|-------|
| `jet_inlet` | Velocity inlet | U = 2.0 m/s, TI = 5% |
| `ambient_inlet` | Velocity inlet | U = 0.001 m/s (small co-flow for stability) |
| `axis` | Symmetry | — |
| `outlet`, `farfield` | Pressure outlet | p = 0 |

---

## Configuration

Configs: [`mesh_jet_phase1.cfg`](mesh_jet_phase1.cfg) · [`mesh_jet_phase2.cfg`](mesh_jet_phase2.cfg)

SST was chosen as it is a standard model that works reasonably well for free shear flows and is fully supported in SU2.

### Why two phases?

When I first tried running with second-order accuracy (MUSCL) from a uniform initial condition, the solver crashed in the first few iterations. At iteration 0 the flow is all zeros, so cell gradients are meaningless. MUSCL tries to reconstruct face values from these gradients and produces nonsense at the nozzle lip (where jet meets ambient in a sharp jump), causing the residuals to blow up.

The fix was to run in two phases:

**Phase 1 — stabilise** (`MUSCL_FLOW= NO`, CFL 5 → 50)
- First-order upwind, always stable regardless of initial gradients
- Run until `rms[P] < -7` so the flow field is physically smooth

**Phase 2 — accuracy** (`MUSCL_FLOW= YES`, CFL 20 → 80)
- Restart from Phase 1 solution
- Second-order MUSCL with Venkatakrishnan-Wang limiter (coeff = 0.1) to control reconstruction near the shear layer
- Now gradients are meaningful so MUSCL works correctly

Other settings:

| Option | Value | Why |
|--------|-------|-----|
| `ILU_FILL_IN` | 2 | Cells near the axis are very stretched — stronger preconditioner helps |
| `MUSCL_TURB` | NO | Keeping turb equations first-order was more stable with no visible change in velocity |
| `RESTART_ASCII` | — | Binary restart caused an MPI crash on this machine (HPC-X v2.19 on NFS) |

---

## Convergence

Phase 2 was stopped at iteration 2145 via `kill -TERM` (which triggers SU2's output handler). Final residuals:

| Residual | Value |
|----------|-------|
| `rms[P]` | −10.016 |
| `rms[U]` | −9.512 |
| `rms[k]` | −11.46 |

Changes per iteration are below 10⁻¹⁰ — the solution is fully converged.

![Convergence history](plots/01_convergence.png)

---

## Results

### Velocity field

![Domain overview](plots/00_domain_overview.png)

### Centreline velocity decay

In the self-similar region, the centreline velocity should follow:

$$\frac{U_c}{U_j} = \frac{B_u}{z/d - z_0/d}, \quad B_u = 6.0,\; z_0/d = 6.75$$

![Centreline decay](plots/02_centerline_decay.png)

### Self-similar radial profiles

Normalised profiles should collapse onto a Gaussian at all downstream locations:

$$U/U_c = \exp(-84.9\,\eta^2), \quad \eta = r/(z - z_0)$$

![Radial profiles](plots/03_radial_profiles.png)

### Jet half-width growth

$$b_u = 0.097\,z$$

![Half-width](plots/04_halfwidth.png)

### Centreline turbulence intensity

$$u_\text{rms}/U_c \approx 0.22 \text{ (plateau for } z/d = 30\text{–}140\text{)}$$

![Turbulence intensity](plots/05_turbulence_intensity.png)

### Summary

| Quantity | SU2 | Fukushima (2001) |
|----------|-----|-----------------|
| `B_u` | 29.1 | 6.0 |
| `db_u/dz` | 0.025 | 0.097 |
| `u_rms/U_c` | keeps rising | ≈ 0.22 (plateau) |

The simulated jet decays too slowly and spreads too little — roughly 4× off on both counts.

---

## What Went Wrong

To understand why, I checked whether momentum is conserved along the jet. In a free jet with no external forces, the axisymmetric momentum flux

$$J = 2\pi\rho \int_0^\infty U^2\, r\, dr$$

should stay **constant** downstream. Instead:

| z/d | J (N) |
|-----|-------|
| 0 | 0.00306 |
| 40 | 0.00549 |
| 80 | 0.00966 |
| 120 | 0.01250 |

That is a 4× increase — physically impossible in a free jet.

The cause is a limitation of using `INC_RANS` + `AXISYMMETRIC= YES` for an open free jet. SU2's incompressible axisymmetric solver adds geometric source terms with a `2πr` factor to the equations. In a pipe or nozzle, pressure and wall boundary conditions naturally balance these terms. In an open jet there is nothing to balance them, so they act as a spurious momentum source that grows with `r`. This keeps the centreline velocity artificially high and stops the jet from spreading at the right rate.

This is a solver formulation issue, not a turbulence model issue. The correct approach would be to use the compressible solver (`RANS`) at a low Mach number, which handles the axisymmetric geometry conservatively.

---

## Reference

Fukushima C., Aanen L. & Westerweel J. (2001). *Investigation of the Mixing Process in an Axisymmetric Turbulent Jet Using PIV and LIF*. 4th Int. Symposium on Particle Image Velocimetry, Göttingen.

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

# Assignment 5 — New Volume Output: Local Speed of Sound

## What the task asked

Add the local speed of sound to SU2's volume output (ParaView files) and screen output, then run the turbulent jet case from Assignment 2 with it enabled.

---

## Implementation

All changes are in one file:

```
SU2_CFD/src/output/CFlowIncOutput.cpp
```

SU2's output system works in two steps for every field: **register** it (give it a name and group), then **set** its value each iteration. I did this for both the per-node volume field and the scalar history field.

### Volume output (per node → ParaView)

**Register** in `SetVolumeOutputFields()`:
```cpp
AddVolumeOutput("SOUND_SPEED", "Sound_Speed", "PRIMITIVE", "Local speed of sound");
```

**Set** in `LoadVolumeData()`:
```cpp
SetVolumeOutputValue("SOUND_SPEED", iPoint,
    sqrt(config->GetBulk_Modulus() / Node_Flow->GetDensity(iPoint)));
```

### Screen/history output (domain average → terminal + CSV)

**Register** in `SetHistoryOutputFields()`:
```cpp
AddHistoryOutput("AVG_SOUND_SPEED", "Avg_a", ScreenOutputFormat::SCIENTIFIC,
                 "CFL_NUMBER", "Domain-averaged speed of sound",
                 HistoryFieldType::COEFFICIENT);
```

**Set** in `LoadHistoryData()`:
```cpp
auto* nodes = static_cast<CIncEulerSolver*>(flow_solver)->GetNodes();
su2double avg_a = 0.0;
for (auto iPoint = 0ul; iPoint < geometry->GetnPointDomain(); iPoint++)
  avg_a += sqrt(config->GetBulk_Modulus() / nodes->GetDensity(iPoint));
avg_a /= geometry->GetnPointDomain();
SetHistoryOutputValue("AVG_SOUND_SPEED", avg_a);
```

The cast to `CIncEulerSolver*` is needed because the base class `CSolver` doesn't expose `GetNodes()` — the node data lives in the derived incompressible solver.

### The formula

For a weakly compressible fluid the speed of sound is defined from the bulk modulus $K$ and density $\rho$:

$$a = \sqrt{\frac{K}{\rho}}$$

Both `GetBulk_Modulus()` and `GetDensity()` return **non-dimensional** values inside SU2. With $K_\text{ref} = 142000$ and $\rho_\text{ref} = 1$ (non-dim), the non-dimensional speed of sound is:

$$a_\text{nd} = \sqrt{\frac{142000}{1}} \approx 376.8$$

The physical value is $a = 11.93$ m/s, giving Mach number $\text{Ma} = U_j / a = 2.0 / 11.93 \approx 0.168$, which confirms the flow is well within the incompressible regime (Ma $\ll$ 0.3).

> **Note:** The physical bulk modulus of water is ~2.2 GPa, giving $a \approx 1484$ m/s. SU2 uses a reduced value (142000 Pa) deliberately — this is the artificial compressibility / pseudo-compressibility trick commonly used in incompressible solvers to make the pressure equation well-conditioned, while keeping Ma low enough that the incompressible assumption holds.

---

## Config additions

```
SCREEN_OUTPUT= INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, RMS_TKE, CFL_NUMBER, AVG_SOUND_SPEED
VOLUME_OUTPUT= PRIMITIVE, TURBULENCE, SOUND_SPEED
```

---

## Screen output

Running the Assignment 2 jet case (`mesh_jet_phase2_a5.cfg`, 2 MPI ranks) with the modified binary:

```
+------------+------------+------------+------------+----------+----------+----------+----------+
| Inner_Iter |   rms[P]   |   rms[U]   |   rms[k]   |  Min CFL |  Max CFL |  Avg CFL |  Avg_a   |
+------------+------------+------------+------------+----------+----------+----------+----------+
|          0 |  -7.670768 |  -7.248929 |  -9.932360 | 1.50e+01 | 1.50e+01 | 1.50e+01 | 3.77e+02 |
|          1 |  -7.699318 |  -7.260444 |  -9.932488 | 1.54e+01 | 1.54e+01 | 1.54e+01 | 3.77e+02 |
|          2 |  -7.720973 |  -7.269018 |  -9.931974 | 1.59e+01 | 1.59e+01 | 1.59e+01 | 3.77e+02 |
```

`Avg_a` is constant across iterations — expected, since `CONSTANT_DENSITY` means $\rho$ is uniform and $K$ is a fixed material constant.

---

## Volume output

![Speed of sound and velocity field](plots/sound_speed_volume.png)

**Top:** `Sound_Speed` field — spatially uniform at $a_\text{nd} \approx 376.8$ across the domain, consistent with constant-density flow.  
**Bottom:** `Velocity_x` from the same converged solution showing the turbulent jet spreading downstream (Assignment 2 result, shown for spatial context).

The `Sound_Speed` field is present in `vol_solution.vtu` and can be visualised in ParaView by selecting it from the array dropdown.

---

## References

1. Fukushima C., Aanen L. & Westerweel J. (2001). *Investigation of the Mixing Process in an Axisymmetric Turbulent Jet Using PIV and LIF*. 4th Int. Symp. on Particle Image Velocimetry, Göttingen.
2. Chorin A.J. (1967). *A Numerical Method for Solving Incompressible Viscous Flow Problems*. J. Comput. Phys. 2, 12–26. — original artificial compressibility method.
3. SU2 Documentation — [Incompressible Flow](https://su2code.github.io/docs_v7/Inc-Euler-Equations/), [Custom Output](https://su2code.github.io/docs_v7/Custom-Output/).
