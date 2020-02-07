module RegisterMismatchCuda

using SharedArrays, Primes, Images, CuArrays, CUDAdrv, CUDAnative
using RegisterCore, RegisterMismatchCommon
using CuArrays.CUFFT
import Base: eltype, ndims
import Images: sdims, coords_spatial, data
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
The major types and functions exported are:

- `mismatch` and `mismatch!`: compute the mismatch between two images
- `mismatch_apertures` and `mismatch_apertures!`: compute apertured mismatch between two images
- `mismatch0`: simple direct mismatch calculation with no shift
- `nanpad`: pad the smaller image with NaNs
- `highpass`: highpass filter an image
- `correctbias!`: replace corrupted mismatch data (due to camera bias inhomogeneity) with imputed data
- `truncatenoise!`: threshold mismatch computation to prevent problems from roundoff
- `aperture_grid`: create a regular grid of apertures
- `allocate_mmarrays`: create storage for output of `mismatch_apertures!`
- `CMStorage`: a type that facilitates re-use of intermediate storage during registration computations
"""
RegisterMismatchCuda

mutable struct CuRCpair{T<:AbstractFloat,N}
    R::CuArray{T,N}
    C::CuArray{Complex{T},N}
    rng::NTuple{N}
end

function CuRCpair{T}(::UndefInitializer, realsize::Dims{N}) where {T<:AbstractFloat,N}
    csize = [realsize...]
    csize[1] = realsize[1]>>1 + 1
    C = CuArray{Complex{T},N}(undef, csize...)
    csize[1] *= 2
    R = reshape(reinterpret(T, vec(C)), (csize...,)::Dims{N})
    rng = map(n->1:n,realsize)
    CuRCpair{T,N}(R, C, rng)
end

function CuRCpair(A::Array{T}) where {T<:AbstractFloat}
    P = CuRCpair(eltype(A), size(A))
    copyto!(P.R, A)
    P
end

function plan_fft_pair(P::CuRCpair{T,N}) where {T,N}
    fwd = CUFFT.plan_rfft(P.R[P.rng...])
    inv = CUFFT.plan_inv(fwd)
    fwd, inv
end

mutable struct NanCorrFFTs{T<:AbstractFloat,N}
    I0::CuRCpair{T,N}
    I1::CuRCpair{T,N}
    I2::CuRCpair{T,N}
end

"""
    CMStorage{T}(undef, aperture_width, maxshift)

Prepare for FFT-based mismatch computations over domains of size `aperture_width`, computing the
mismatch up to shifts of size `maxshift`.
"""
mutable struct CMStorage{T<:AbstractFloat,N}
    aperture_width::Vector{Float64}
    maxshift::Vector{Int}
    getindexes::Vector{UnitRange{Int}}   # indexes for pulling padded data, in source-coordinates
    setindexes::Vector{UnitRange{Int}}   # indexes for pushing fixed data, in source-coordinates
    fixed::NanCorrFFTs{T,N}
    moving::NanCorrFFTs{T,N}
    num::CuRCpair{T,N}
    denom::CuRCpair{T,N}
    numhost::Array{T,N}
    denomhost::Array{T,N}
    # the next two store the result of calling plan_fft! and plan_ifft!
    fftfunc::Any
    ifftfunc::Any
    shiftindices::Vector{Vector{Int}} # indices for performing fftshift & snipping from -maxshift:maxshift
end

function CMStorage{T,N}(::UndefInitializer, aperture_width::NTuple{N,<:Real}, maxshift::Dims{N}) where {T<:AbstractFloat,N}
    blocksize = map(x->ceil(Int,x), aperture_width)
    padsz = padsize(blocksize, maxshift)
    getindexes = padranges(blocksize, maxshift)
    setindexes = UnitRange{Int}[(1:blocksize[i]).+maxshift[i] for i = 1:length(blocksize)]
    fixed  = NanCorrFFTs(CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz))
    moving = NanCorrFFTs(CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz), CuRCpair{T}(undef, padsz))
    num = CuRCpair{T}(undef, padsz)
    denom = CuRCpair{T}(undef, padsz)
    mmsz = map(x->2x+1, (maxshift...,))
    numhost, denomhost = Array{T}(undef, mmsz), Array{T}(undef, mmsz)
    fftfunc, ifftfunc = plan_fft_pair(num)
    maxshiftv = [maxshift...]
    shiftindices = Vector{Int}[ [padsz[i].+(-maxshift[i]+1:0); 1:maxshift[i]+1] for i = 1:length(maxshift) ]
    CMStorage{T,N}(Float64[aperture_width...], maxshiftv, getindexes, setindexes, fixed, moving, num, denom, numhost, denomhost, fftfunc, ifftfunc, shiftindices)
end

CMStorage{T}(::UndefInitializer, aperture_width::NTuple{N,<:Real}, maxshift::Dims{N}) where {T<:AbstractFloat,N} =
    CMStorage{T,N}(undef, aperture_width, maxshift)

# Note: display doesn't do anything, but here for compatibility with the CPU version
CMStorage{T}(::UndefInitializer, blocksize::NTuple{N,<:Real}, maxshift::Dims{N}; display=false) where {T<:Real,N} = CMStorage{T}(undef, blocksize, maxshift)
CMStorage{T,N}(::UndefInitializer, blocksize::NTuple{N,<:Real}, maxshift::Dims{N}; display=false) where {T<:Real,N} = CMStorage{T,N}(undef, blocksize, maxshift)

context(cms::CMStorage) = context(cms.num.C)
context(a::CuArray) = a.ctx

eltype(cms::CMStorage{T,N}) where {T,N} = T
 ndims(cms::CMStorage{T,N}) where {T,N} = N

# Some tools from Images
sdims(A::CuArray) = ndims(A)
coords_spatial(A::CuArray) = 1:ndims(A)

### Main API

"""
`mm = mismatch([T], fixed, moving, maxshift;
[normalization=:intensity])` computes the mismatch between `fixed` and
`moving` as a function of translations (shifts) up to size `maxshift`.
Optionally specify the element-type of the mismatch arrays (default
`Float32` for Integer- or FixedPoint-valued images) and the
normalization scheme (`:intensity` or `:pixels`).

`fixed` and `moving` must have the same size; you can pad with
`NaN`s as needed. See `nanpad`.

This operation is synchronous with respect to the host.
"""
function mismatch(::Type{T}, fixed::AbstractArray, moving::AbstractArray, maxshift::DimsLike; normalization = :intensity) where T<:Real
    assertsamesize(fixed, moving)
    d_fixed  = CuArray{T}(fixed)
    d_moving = CuArray{T}(moving)
    mm = mismatch(d_fixed, d_moving, maxshift, normalization=normalization)
    mm
end

function mismatch(fixed::CuArray{T}, moving::CuArray{T}, maxshift::DimsLike; normalization = :intensity) where T
    assertsamesize(fixed, moving)
    nd = ndims(fixed)
    cms = CMStorage{T}(undef, size(fixed), maxshift)
    mm = MismatchArray(T, (2 .* maxshift .+ 1)...)
    fillfixed!(cms, fixed)
    mismatch!(mm, cms, moving, normalization=normalization)
    mm
end

"""
`mms = mismatch_apertures([T], fixed, moving, gridsize, maxshift;
[normalization=:pixels], [flags=FFTW.MEASURE], kwargs...)` computes
the mismatch between `fixed` and `moving` over a regularly-spaced grid
of aperture centers, effectively breaking the images up into
chunks. The maximum-allowed shift in any aperture is `maxshift`.

`mms = mismatch_apertures([T], fixed, moving, aperture_centers,
aperture_width, maxshift; kwargs...)` computes the mismatch between
`fixed` and `moving` over a list of apertures of size `aperture_width`
at positions defined by `aperture_centers`.

`fixed` and `moving` must have the same size; you can pad with `NaN`s
as needed to ensure this.  You can optionally specify the real-valued
element type mm; it defaults to the element type of `fixed` and
`moving` or, for Integer- or FixedPoint-valued images, `Float32`.

On output, `mms` will be an Array-of-MismatchArrays, with the outer
array having the same "grid" shape as `aperture_centers`.  The centers
can in general be provided as an vector-of-tuples, vector-of-vectors,
or a matrix with each point in a column.  If your centers are arranged
in a rectangular grid, you can use an `N`-dimensional array-of-tuples
(or array-of-vectors) or an `N+1`-dimensional array with the center
positions specified along the first dimension. See `aperture_grid`.
"""
function mismatch_apertures(::Type{T},
                            fixed::AbstractArray,
                            moving::AbstractArray,
                            aperture_centers::AbstractArray,
                            aperture_width::WidthLike,
                            maxshift::DimsLike;
                            kwargs...) where T
    assertsamesize(fixed, moving)
    d_fixed  = CuArray{T}(sdata(fixed))
    d_moving = CuArray{T}(moving)
    mms = mismatch_apertures(d_fixed, d_moving, aperture_centers, aperture_width, maxshift; kwargs...)
    mms
end

# only difference here relative to RegisterMismatch is the lack of the
# FFTW keywords
function mismatch_apertures(fixed::CuArray{T},
                            moving::CuArray,
                            aperture_centers::AbstractArray,
                            aperture_width::WidthLike,
                            maxshift::DimsLike;
                            normalization = :pixels,
                            kwargs...) where T
    nd = sdims(fixed)
    assertsamesize(fixed,moving)
    (length(aperture_width) == nd && length(maxshift) == nd) || error("Dimensionality mismatch")
    mms = allocate_mmarrays(T, aperture_centers, maxshift)
    cms = CMStorage{T}(undef, aperture_width, maxshift; kwargs...)
    mismatch_apertures!(mms, fixed, moving, aperture_centers, cms; normalization=normalization)
    mms
end

function fillfixed!(cms::CMStorage{T}, fixed::CuArray; f_indexes = ntuple(i->1:size(fixed,i), ndims(fixed))) where T
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
    for idim = 1:nd
        tmp = f_indexes[idim]
        i1 = first(tmp) >= 1 ? 1 : 2-first(tmp)
        i2 = last(tmp) <= size(fixed, idim) ? length(tmp) : length(tmp)-(last(tmp)-size(fixed, idim))
        srcindexes[idim] = tmp[i1]:tmp[i2]
        dstindexes[idim] = cms.setindexes[idim][i1]:cms.setindexes[idim][i2]
    end
    copyto!(paddedf, tuple(dstindexes...), fixed, tuple(srcindexes...))
    # Prepare the components of the convolution
    threadspb = calculate_threads(size(paddedf), attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)÷2)
    nblocks = ceil.(Int, size(paddedf)./ threadspb)
    @cuda blocks = nblocks threads = threadspb kernel_conv_components!(paddedf, cms.fixed.I2.R, cms.fixed.I0.R)
    synchronize()
    # Compute FFTs
    obj = cms.fixed
    for item in (obj.I0, obj.I1, obj.I2)
        copyto!(item.C, cms.fftfunc * item.R[item.rng...])
    end
    obj
end

"""
`mismatch!(mm, cms, moving; [normalization=:intensity])`
computes the mismatch as a function of shift, storing the result in
`mm`. The `fixed` image has been prepared in `cms`, a `CMStorage` object.
"""
function mismatch!(mm::MismatchArray, cms::CMStorage{T}, moving::CuArray; normalization = :intensity, m_offset = ntuple(i->0, ndims(cms))) where T
    ctx = context(cms)
    context(moving) == ctx || error("Fixed and cms must be on the same context")
    activate(ctx)
    dev = device(ctx)
    checksize_maxshift(mm, cms.maxshift)
    nd = ndims(cms)
    paddedm = cms.moving.I1.R
    get!(paddedm, moving, ntuple(d->cms.getindexes[d].+m_offset[d], nd), T(NaN))
    # Prepare the components of the convolution
    threadspb = calculate_threads(size(paddedm), attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)÷2)
    nblocks = ceil.(Int, size(paddedm)./ threadspb)
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
    args = (cms.fixed.I1.C,  cms.fixed.I2.C,  cms.fixed.I0.C,
            cms.moving.I1.C, cms.moving.I2.C, cms.moving.I0.C,
            cms.num.C, cms.denom.C)
    threadspb = calculate_threads(size(d_numC), attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)÷2)
    nblocks = ceil.(Int, size(d_numC)./ threadspb)
    if normalization == :intensity
        @cuda blocks = nblocks threads = threadspb kernel_calcNumDenom_intensity!(args...)
    elseif normalization == :pixels
        @cuda blocks = nblocks threads = threadspb kernel_calcNumDenom_pixels!(args...)
    else
        throw(ArgumentError("normalizeby=$(normalizeby) not recognized"))
    end
    synchronize()

    # Compute the IFFTs
    d_num = cms.num.R
    d_denom = cms.denom.R
    rng = cms.num.rng
    copyto!(d_num, rng, cms.ifftfunc * d_numC, rng)
    copyto!(d_denom, rng, cms.ifftfunc * d_denomC, rng)
    # Copy result to host
    copyto!(mm, (view(Array(d_num), cms.shiftindices...), view(Array(d_denom), cms.shiftindices...)))
end

"""
`mismatch_apertures!(mms, fixed, moving, aperture_centers, cms;
[normalization=:pixels])` computes the mismatch between `fixed` and
`moving` over a list of apertures at positions defined by
`aperture_centers`.  The parameters and working storage are contained
in `cms`, a `CMStorage` object. The results are stored in `mms`, an
Array-of-MismatchArrays which must have length equal to the number of
aperture centers.
"""
function mismatch_apertures!(mms, fixed, moving, aperture_centers, cms; normalization=:pixels)
    assertsamesize(fixed, moving)
    N = ndims(cms)
    for (mm,center) in zip(mms, each_point(aperture_centers))
        rng = aperture_range(center, cms.aperture_width)
        fillfixed!(cms, fixed; f_indexes=rng)
        offset = [first(rng[d])-1 for d = 1:N]
        mismatch!(mm, cms, moving; normalization=normalization, m_offset=offset)
    end
    synchronize()
    mms
end


### Utilities

function assertsamesize(A::AbstractArray, B::AbstractArray)
    size(A,1) == size(B,1) && size(A,2) == size(B,2) && size(A,3) == size(B,3) || error("Arrays are not the same size")
end

### Deprecations

@deprecate CuRCpair(realtype::Type{T}, realsize) where {T<:AbstractFloat} CuRCpair{T}(undef, realsize)

function CMStorage{T}(::UndefInitializer, aperture_width::WidthLike, maxshift::DimsLike; kwargs...) where {T<:Real}
    Base.depwarn("CMStorage with aperture_width::$(typeof(aperture_width)) and maxshift::$(typeof(maxshift)) is deprecated, use tuples instead", :CMStorage)
    (N = length(aperture_width)) == length(maxshift) || error("Dimensionality mismatch")
    return CMStorage{T,N}(undef, (aperture_width...,), (maxshift...,); kwargs...)
end

@deprecate CMStorage(::Type{T}, aperture_width::WidthLike, maxshift::DimsLike; kwargs...) where {T<:AbstractFloat}  CMStorage{T}(undef, aperture_width, maxshift; kwargs...)

end # module
