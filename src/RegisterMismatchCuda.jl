module RegisterMismatchCuda

using CUDA: CUDA, CUFFT, @cuda, CuArray, activate, attribute, blockDim, blockIdx, context, device,
    synchronize, threadIdx
using ImageCore: ImageCore
using Primes: Primes, factor
using RegisterCore: RegisterCore, MismatchArray, maxshift
using RegisterMismatchCommon: RegisterMismatchCommon, DimsLike, WidthLike, allocate_mmarrays,
    aperture_range, checksize_maxshift, each_point, padranges, padsize
using SharedArrays: SharedArrays, sdata
import Base: eltype, ndims
import RegisterMismatchCommon: mismatch, mismatch_apertures, mismatch0

export
    CMStorage,
    fillfixed!,
    mismatch0,
    mismatch,
    mismatch!,
    mismatch_apertures,
    mismatch_apertures!,
    CuRCpair

include("kernels.jl")

"""
Provide GPU-accelerated mismatch computation for image registration using CUDA.

The major types and functions exported are:

- `mismatch` and `mismatch!`: compute the mismatch between two images as a function of shift
- `mismatch_apertures` and `mismatch_apertures!`: compute mismatch over localized apertures
- `mismatch0`: simple direct mismatch with no shift
- `allocate_mmarrays`: create storage for the output of `mismatch_apertures!`
- `CMStorage`: pre-allocated working storage for repeated mismatch computations
- `CuRCpair`: paired real/complex `CuArray` used for in-place FFT operations
- `fillfixed!`: load and FFT the fixed image into a `CMStorage` object
"""
RegisterMismatchCuda

"""
    CuRCpair{T}(undef, realsize::Dims{N})
    CuRCpair(A::AbstractArray{T})

A paired real/complex `CuArray` sharing GPU memory, used for in-place FFT computations.

`CuRCpair{T}(undef, realsize)` allocates a real array `R` of size `realsize` and a
complex array `C` of size `(realsize[1]÷2+1, realsize[2:end]...)` that share the same
underlying GPU memory (matching the layout of a real-to-complex FFT plan). The field `rng`
holds `UnitRange`s indexing the unpadded region of `R`.

`CuRCpair(A::AbstractArray{T})` uploads the host array `A` into the real part of a newly
allocated pair.

Fields:
- `R`: real-valued `CuArray` (with in-place FFT padding along the first dimension)
- `C`: complex-valued `CuArray` sharing memory with `R`
- `rng`: `NTuple{N, UnitRange{Int}}` indexing the unpadded region of `R`
"""
struct CuRCpair{T <: AbstractFloat, N}
    R::CuArray{T, N}
    C::CuArray{Complex{T}, N}
    rng::NTuple{N}
end

function CuRCpair{T}(::UndefInitializer, realsize::Dims{N}) where {T <: AbstractFloat, N}
    csize = [realsize...]
    csize[1] = realsize[1] >> 1 + 1
    C = CuArray{Complex{T}, N}(undef, csize...)
    csize[1] *= 2
    R = reshape(reinterpret(T, vec(C)), (csize...,)::Dims{N})
    rng = map(n -> 1:n, realsize)
    return CuRCpair{T, N}(R, C, rng)
end

function CuRCpair(A::AbstractArray{T}) where {T <: AbstractFloat}
    P = CuRCpair{T}(undef, size(A))
    copyto!(view(P.R, P.rng...), A)
    return P
end

function plan_fft_pair(P::CuRCpair{T, N}) where {T, N}
    fwd = CUFFT.plan_rfft(P.R[P.rng...])
    inv = CUFFT.plan_inv(fwd)
    return fwd, inv
end

struct NanCorrFFTs{T <: AbstractFloat, N}
    I0::CuRCpair{T, N}
    I1::CuRCpair{T, N}
    I2::CuRCpair{T, N}
end

"""
    CMStorage{T}(undef, aperture_width::NTuple{N,<:Real}, maxshift::Dims{N})
    CMStorage{T,N}(undef, aperture_width::NTuple{N,<:Real}, maxshift::Dims{N})

Pre-allocate GPU working storage for FFT-based mismatch computations.

`aperture_width` is an `N`-tuple specifying the image region size, and `maxshift` is an
`N`-tuple specifying the maximum shift to evaluate in each dimension. `T` sets the
floating-point precision.

The typical workflow is:
1. Construct `cms = CMStorage{T}(undef, aperture_width, maxshift)`
2. Call `fillfixed!(cms, fixed)` to load and FFT the fixed image
3. Call `mismatch!(mm, cms, moving)` for each moving image

# Examples
```julia
cms = CMStorage{Float32}(undef, (64, 64), (10, 10))
```
"""
struct CMStorage{T <: AbstractFloat, N}
    aperture_width::Vector{Float64}
    maxshift::Vector{Int}
    getindexes::Vector{UnitRange{Int}}   # indexes for pulling padded data, in source-coordinates
    setindexes::Vector{UnitRange{Int}}   # indexes for pushing fixed data, in source-coordinates
    fixed::NanCorrFFTs{T, N}
    moving::NanCorrFFTs{T, N}
    num::CuRCpair{T, N}
    denom::CuRCpair{T, N}
    numhost::Array{T, N}
    denomhost::Array{T, N}
    # the next two store the result of calling plan_fft! and plan_ifft!
    fftfunc::Any
    ifftfunc::Any
    shiftindices::Vector{Vector{Int}} # indices for performing fftshift & snipping from -maxshift:maxshift
end

function CMStorage{T, N}(::UndefInitializer, aperture_width::NTuple{N, <:Real}, maxshift::Dims{N}) where {T <: AbstractFloat, N}
    blocksize = map(x -> ceil(Int, x), aperture_width)
    padsz = padsize(blocksize, maxshift)
    getindexes = padranges(blocksize, maxshift)
    setindexes = UnitRange{Int}[(1:blocksize[i]) .+ maxshift[i] for i in 1:length(blocksize)]
    fixed = NanCorrFFTs(CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz))
    moving = NanCorrFFTs(CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz))
    num = CuRCpair{T}(undef, padsz)
    denom = CuRCpair{T}(undef, padsz)
    mmsz = map(x -> 2x + 1, (maxshift...,))
    numhost, denomhost = Array{T}(undef, mmsz), Array{T}(undef, mmsz)
    fftfunc, ifftfunc = plan_fft_pair(num)
    maxshiftv = [maxshift...]
    shiftindices = Vector{Int}[ [padsz[i] .+ ((-maxshift[i] + 1):0); 1:(maxshift[i] + 1)] for i in 1:length(maxshift) ]
    return CMStorage{T, N}(Float64[aperture_width...], maxshiftv, getindexes, setindexes, fixed, moving, num, denom, numhost, denomhost, fftfunc, ifftfunc, shiftindices)
end

CMStorage{T}(::UndefInitializer, aperture_width::NTuple{N, <:Real}, maxshift::Dims{N}) where {T <: AbstractFloat, N} =
    CMStorage{T, N}(undef, aperture_width, maxshift)

# Note: display doesn't do anything, but here for compatibility with the CPU version
CMStorage{T}(::UndefInitializer, blocksize::NTuple{N, <:Real}, maxshift::Dims{N}; display = false) where {T <: Real, N} = CMStorage{T}(undef, blocksize, maxshift)
CMStorage{T, N}(::UndefInitializer, blocksize::NTuple{N, <:Real}, maxshift::Dims{N}; display = false) where {T <: Real, N} = CMStorage{T, N}(undef, blocksize, maxshift)

CUDA.context(cms::CMStorage) = context(cms.num.C)

eltype(cms::CMStorage{T, N}) where {T, N} = T
ndims(cms::CMStorage{T, N}) where {T, N} = N

### Main API

"""
    mismatch([T], fixed, moving, maxshift; normalization=:intensity) -> MismatchArray
    mismatch(fixed::CuArray{T}, moving::CuArray{T}, maxshift; normalization=:intensity) -> MismatchArray

Compute the mismatch between `fixed` and `moving` as a function of translations up to
size `maxshift`.

The first form accepts host arrays and uploads them to the GPU internally; the second
operates on arrays already resident on the GPU. Optionally specify the element type `T`
(default `Float32` for integer- or fixed-point images) and the `normalization` scheme
(`:intensity` or `:pixels`).

`fixed` and `moving` must have the same size. Pad with `NaN`s to equalize sizes if needed.

Returns a `MismatchArray` of size `(2*maxshift[i]+1 for i in 1:N)`. This operation is
synchronous with respect to the host.
"""
function mismatch(::Type{T}, fixed::AbstractArray, moving::AbstractArray, maxshift::DimsLike; normalization = :intensity) where {T <: Real}
    assertsamesize(fixed, moving)
    d_fixed = CuArray{T}(fixed)
    d_moving = CuArray{T}(moving)
    mm = mismatch(d_fixed, d_moving, maxshift, normalization = normalization)
    return mm
end

function mismatch(fixed::CuArray{T}, moving::CuArray{T}, maxshift::DimsLike; normalization = :intensity) where {T}
    assertsamesize(fixed, moving)
    nd = ndims(fixed)
    cms = CMStorage{T}(undef, size(fixed), maxshift)
    mm = MismatchArray(T, (2 .* maxshift .+ 1)...)
    fillfixed!(cms, fixed)
    mismatch!(mm, cms, moving, normalization = normalization)
    return mm
end

"""
    mismatch_apertures([T], fixed, moving, aperture_centers, aperture_width, maxshift;
                        normalization=:pixels, kwargs...) -> Array{MismatchArray}

Compute the mismatch between `fixed` and `moving` over a list of localized apertures.

Each aperture is centered at a point in `aperture_centers` and has size `aperture_width`.
The maximum allowed shift within each aperture is `maxshift`. Optionally specify the
element type `T` (default: element type of `fixed` and `moving`, or `Float32` for
integer/fixed-point images).

`fixed` and `moving` must have the same size. Pad with `NaN`s as needed.

`aperture_centers` can be a vector-of-tuples, vector-of-vectors, or an `N`-dimensional
array-of-tuples for grid layouts. See `aperture_grid` from `RegisterMismatchCommon` for
constructing regular grids.

Returns an `Array{MismatchArray}` with the same shape as `aperture_centers`.
"""
function mismatch_apertures(
        ::Type{T},
        fixed::AbstractArray,
        moving::AbstractArray,
        aperture_centers::AbstractArray,
        aperture_width::WidthLike,
        maxshift::DimsLike;
        kwargs...
    ) where {T}
    assertsamesize(fixed, moving)
    d_fixed = CuArray{T}(sdata(fixed))
    d_moving = CuArray{T}(moving)
    mms = mismatch_apertures(d_fixed, d_moving, aperture_centers, aperture_width, maxshift; kwargs...)
    return mms
end

# only difference here relative to RegisterMismatch is the lack of the
# FFTW keywords
function mismatch_apertures(
        fixed::CuArray{T},
        moving::CuArray,
        aperture_centers::AbstractArray,
        aperture_width::WidthLike,
        maxshift::DimsLike;
        normalization = :pixels,
        kwargs...
    ) where {T}
    nd = ndims(fixed)
    assertsamesize(fixed, moving)
    (length(aperture_width) == nd && length(maxshift) == nd) || error("Dimensionality mismatch")
    mms = allocate_mmarrays(T, aperture_centers, maxshift)
    cms = CMStorage{T}(undef, aperture_width, maxshift; kwargs...)
    mismatch_apertures!(mms, cms, fixed, moving, aperture_centers; normalization = normalization)
    return mms
end

"""
    fillfixed!(cms::CMStorage, fixed::CuArray;
               f_indexes=ntuple(i->1:size(fixed,i), ndims(fixed))) -> NanCorrFFTs

Load `fixed` into the pre-allocated storage `cms`, computing all FFTs needed for
subsequent `mismatch!` calls.

The optional keyword `f_indexes` is an `N`-tuple of `UnitRange`s selecting a
sub-region of `fixed` to use (default: the full array). Ranges extending beyond
the array bounds are clamped; out-of-bounds regions are padded with `NaN`.

`fixed` and `cms` must reside on the same CUDA context. After calling `fillfixed!`,
call `mismatch!` with a moving image to compute the mismatch result.
"""
function fillfixed!(cms::CMStorage{T}, fixed::CuArray; f_indexes = ntuple(i -> 1:size(fixed, i), ndims(fixed))) where {T}
    ctx = context(cms)
    context(fixed) == ctx || error("Fixed and cms must be on the same context")
    nd = ndims(cms)
    ndims(fixed) == nd || error("Fixed and cms must have the same dimensionality")
    activate(ctx)
    dev = device(ctx)
    # Pad
    paddedf = cms.fixed.I1.R
    fill!(paddedf, NaN)
    dstindexes = Array{UnitRange{Int}}(undef, nd)
    srcindexes = Array{UnitRange{Int}}(undef, nd)
    for idim in 1:nd
        tmp = f_indexes[idim]
        i1 = first(tmp) >= 1 ? 1 : 2 - first(tmp)
        i2 = last(tmp) <= size(fixed, idim) ? length(tmp) : length(tmp) - (last(tmp) - size(fixed, idim))
        srcindexes[idim] = tmp[i1]:tmp[i2]
        dstindexes[idim] = cms.setindexes[idim][i1]:cms.setindexes[idim][i2]
    end
    copyto!(paddedf, CartesianIndices(Tuple(dstindexes)), fixed, CartesianIndices(Tuple(srcindexes))) #This conversion may be inefficient.
    # Prepare the components of the convolution
    threadspb = calculate_threads(size(paddedf), attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) ÷ 2)
    nblocks = ceil.(Int, size(paddedf) ./ threadspb)
    @cuda blocks = nblocks threads = threadspb kernel_conv_components!(paddedf, cms.fixed.I2.R, cms.fixed.I0.R)
    synchronize()
    # Compute FFTs
    obj = cms.fixed
    for item in (obj.I0, obj.I1, obj.I2)
        copyto!(item.C, cms.fftfunc * item.R[item.rng...])
    end
    return obj
end

"""
    mismatch!(mm, cms, moving; normalization=:intensity,
              m_offset=ntuple(i->0, ndims(cms))) -> MismatchArray

Compute the mismatch as a function of shift, storing the result in `mm`.

`cms` is a `CMStorage` that has been loaded with the fixed image via `fillfixed!`.
`moving` is the GPU array to compare against. `normalization` selects the
normalization scheme (`:intensity` or `:pixels`). `m_offset` is an `N`-tuple of
integer offsets applied to the moving image window, shifting which region of
`moving` is compared.

Returns `mm`.
"""
function mismatch!(mm::MismatchArray, cms::CMStorage{T}, moving::CuArray; normalization = :intensity, m_offset = ntuple(i -> 0, ndims(cms))) where {T}
    ctx = context(cms)
    context(moving) == ctx || error("Fixed and cms must be on the same context")
    activate(ctx)
    dev = device(ctx)
    checksize_maxshift(mm, cms.maxshift)
    nd = ndims(cms)
    paddedm = cms.moving.I1.R
    get!(paddedm, moving, ntuple(d -> cms.getindexes[d] .+ m_offset[d], nd), T(NaN))
    # Prepare the components of the convolution
    threadspb = calculate_threads(size(paddedm), attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) ÷ 2)
    nblocks = ceil.(Int, size(paddedm) ./ threadspb)
    @cuda blocks = nblocks threads = threadspb kernel_conv_components!(paddedm, cms.moving.I2.R, cms.moving.I0.R)
    synchronize()
    # Compute FFTs
    obj = cms.moving
    for item in (obj.I0, obj.I1, obj.I2)
        copyto!(item.C, cms.fftfunc * item.R[item.rng...])
    end
    # Perform the convolution in fourier space
    d_numC = cms.num.C
    d_denomC = cms.denom.C
    args = (
        cms.fixed.I1.C, cms.fixed.I2.C, cms.fixed.I0.C,
        cms.moving.I1.C, cms.moving.I2.C, cms.moving.I0.C,
        cms.num.C, cms.denom.C,
    )
    threadspb = calculate_threads(size(d_numC), attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) ÷ 2)
    nblocks = ceil.(Int, size(d_numC) ./ threadspb)
    if normalization == :intensity
        @cuda blocks = nblocks threads = threadspb kernel_calcNumDenom_intensity!(args...)
    elseif normalization == :pixels
        @cuda blocks = nblocks threads = threadspb kernel_calcNumDenom_pixels!(args...)
    else
        throw(ArgumentError("normalization=$(normalization) not recognized"))
    end
    synchronize()

    # Compute the IFFTs
    d_num = cms.num.R
    d_denom = cms.denom.R
    rng = CartesianIndices(cms.num.rng)
    copyto!(d_num, rng, cms.ifftfunc * d_numC, rng)
    copyto!(d_denom, rng, cms.ifftfunc * d_denomC, rng)
    # Copy result to host
    return copyto!(mm, (view(Array(d_num), cms.shiftindices...), view(Array(d_denom), cms.shiftindices...)))
end

"""
    mismatch_apertures!(mms, cms, fixed, moving, aperture_centers;
                        normalization=:pixels) -> Array{MismatchArray}

Compute the mismatch between `fixed` and `moving` over a list of apertures, storing
results in the pre-allocated `mms`.

`mms` must be an `Array{MismatchArray}` with the same number of elements as
`aperture_centers`, typically created by `allocate_mmarrays`. `cms` is a `CMStorage`
object whose `aperture_width` and `maxshift` determine the aperture size and shift range.

Returns `mms`.

# Examples
```julia
aperture_centers = [(32, 32), (32, 96), (96, 32), (96, 96)]
maxshift = (10, 10)
aperture_width = (64, 64)
cms = CMStorage{Float32}(undef, aperture_width, maxshift)
mms = allocate_mmarrays(Float32, aperture_centers, maxshift)
d_fixed = CuArray(rand(Float32, 128, 128))
d_moving = CuArray(rand(Float32, 128, 128))
mismatch_apertures!(mms, cms, d_fixed, d_moving, aperture_centers)
```
"""
function mismatch_apertures!(
        mms::AbstractArray{<:MismatchArray},
        cms::CMStorage,
        fixed::CuArray,
        moving::CuArray,
        aperture_centers::AbstractArray;
        normalization = :pixels,
    )
    assertsamesize(fixed, moving)
    N = ndims(cms)
    for (mm, center) in zip(mms, each_point(aperture_centers))
        rng = aperture_range(center, cms.aperture_width)
        fillfixed!(cms, fixed; f_indexes = rng)
        offset = [first(rng[d]) - 1 for d in 1:N]
        mismatch!(mm, cms, moving; normalization = normalization, m_offset = offset)
    end
    synchronize()
    return mms
end

# Deprecated argument order: (mms, fixed, moving, aperture_centers, cms)
function mismatch_apertures!(mms, fixed, moving, aperture_centers, cms::CMStorage; normalization = :pixels)
    Base.depwarn(
        "`mismatch_apertures!(mms, fixed, moving, aperture_centers, cms)` is deprecated. " *
        "Use `mismatch_apertures!(mms, cms, fixed, moving, aperture_centers)` instead.",
        :mismatch_apertures!,
    )
    return mismatch_apertures!(mms, cms, fixed, moving, aperture_centers; normalization)
end


### Utilities

function assertsamesize(A::AbstractArray, B::AbstractArray)
    return size(A, 1) == size(B, 1) && size(A, 2) == size(B, 2) && size(A, 3) == size(B, 3) || error("Arrays are not the same size")
end

end # module
