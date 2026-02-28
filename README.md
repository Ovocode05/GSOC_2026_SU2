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
