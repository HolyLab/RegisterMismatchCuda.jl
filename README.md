# RegisterMismatchCuda

[![Build Status](https://github.com/HolyLab/RegisterMismatchCuda.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/HolyLab/RegisterMismatchCuda.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/HolyLab/RegisterMismatchCuda.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterMismatchCuda.jl)

GPU-accelerated mismatch computation for image registration. This package is a
CUDA backend for the [Register](https://github.com/HolyLab) image-registration
ecosystem, implementing the same interface as
[RegisterMismatch](https://github.com/HolyLab/RegisterMismatch.jl) but running
on the GPU via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

## What is "mismatch"?

Mismatch quantifies how well two images align at a given pixel shift. For a
shift **d**, it is the sum of squared differences between the fixed image and
the shifted moving image, divided by a normalization factor. The shift with the
smallest mismatch is the best-fit translation between the images.

Each result entry is stored as a `(numerator, denominator)` pair inside a
[`MismatchArray`](https://github.com/HolyLab/RegisterCore.jl). Use
`RegisterCore.separate` to extract the pair and
`RegisterCore.indmin_mismatch(mm, thresh)` to find the optimal shift while
ignoring poorly-normalized entries.

## Installation

This package lives in the [HolyLab registry](https://github.com/HolyLab/HolyLabRegistry).
Add the registry once, then install as usual:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterMismatchCuda")
```

A working CUDA installation and a compatible GPU are required at runtime.
See [CUDA.jl setup](https://cuda.juliagpu.org/stable/installation/overview/) for details.

## Usage

### Simple one-shot mismatch

Compare two images up to a given maximum shift. Host arrays are uploaded to the
GPU automatically:

```julia
using RegisterMismatchCuda, RegisterCore

fixed  = rand(Float32, 256, 256)
moving = rand(Float32, 256, 256)
maxshift = (20, 20)

mm = mismatch(fixed, moving, maxshift)        # returns a MismatchArray
best_shift = RegisterCore.indmin_mismatch(mm, 0.01)
```

`best_shift` is a `CartesianIndex` giving the translation (in pixels) from
`fixed` to `moving`, indexed relative to zero (e.g. `CartesianIndex(-3, 5)`
means the moving image is shifted 3 pixels up and 5 pixels to the right).

Pass `normalization=:pixels` (default is `:intensity`) to normalize by the
number of overlapping pixels rather than by image intensity.

### Repeated comparisons against a fixed image (CMStorage workflow)

When comparing many moving images against a single fixed image, pre-allocating
`CMStorage` avoids redundant GPU allocations and re-computing the fixed image's
FFTs:

```julia
using RegisterMismatchCuda, RegisterCore, CUDA

aperture_width = (256, 256)
maxshift = (20, 20)

cms = CMStorage{Float32}(undef, aperture_width, maxshift)

d_fixed  = CuArray(rand(Float32, 256, 256))
d_moving = CuArray(rand(Float32, 256, 256))
mm = RegisterCore.MismatchArray(Float32, (2 .* maxshift .+ 1)...)

fillfixed!(cms, d_fixed)            # compute fixed-image FFTs once

mismatch!(mm, cms, d_moving)        # fast: only moving-image FFTs + correlation
best_shift = RegisterCore.indmin_mismatch(mm, 0.01)

# reuse cms for another moving image without touching d_fixed again
d_moving2 = CuArray(rand(Float32, 256, 256))
mismatch!(mm, cms, d_moving2)
```

### Localized (aperture) mismatch

For non-rigid or spatially varying registration, compute mismatch independently
inside a grid of localized apertures. Each aperture produces its own
`MismatchArray`:

```julia
using RegisterMismatchCuda, RegisterMismatchCommon, RegisterCore

fixed  = rand(Float32, 512, 512)
moving = rand(Float32, 512, 512)

# 4×4 grid of apertures (centers and widths chosen automatically), max shift ±15
gridsize = (4, 4)
maxshift = (15, 15)

mms = mismatch_apertures(Float32, fixed, moving, gridsize, maxshift)

# mms is an Array{MismatchArray} with shape gridsize
# find the best shift at each aperture location:
shifts = map(mm -> RegisterCore.indmin_mismatch(mm, 0.01), mms)
```

For irregular aperture layouts, pass explicit aperture centers and an explicit
width instead:

```julia
aperture_centers = RegisterMismatchCommon.aperture_grid(size(fixed), gridsize)
aperture_width   = RegisterMismatchCommon.default_aperture_width(fixed, gridsize)
mms = mismatch_apertures(Float32, fixed, moving, aperture_centers, aperture_width, maxshift)
```
