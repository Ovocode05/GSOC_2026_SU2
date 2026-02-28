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
