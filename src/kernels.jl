cucartesianindex(a::AbstractVector) = CartesianIndex((blockIdx().x-1) * blockDim().x + threadIdx().x)
cucartesianindex(a::AbstractMatrix) = CartesianIndex((blockIdx().x-1) * blockDim().x + threadIdx().x,
                                                     (blockIdx().y-1) * blockDim().y + threadIdx().y)
cucartesianindex(a::AbstractArray{<:Any,3}) = CartesianIndex((blockIdx().x-1) * blockDim().x + threadIdx().x,
                                                             (blockIdx().y-1) * blockDim().y + threadIdx().y,
                                                             (blockIdx().z-1) * blockDim().z + threadIdx().z)

function calculate_threads(sz::Dims{N}, maxthreadspb) where N
    threads = ones(Int, N)
    L = prod(sz)
    iterdim = Iterators.cycle(1:N)
    dim, state = iterate(iterdim)
    for (b, p) in factor(maxthreadspb)
        for i = 1:p
            P = prod(threads)
            P >= L && return (threads...,)::Dims{N}
            threads[dim] *= b
            dim, state = iterate(iterdim, state)
        end
    end
    return (threads...,)::Dims{N}
end

#= CUDA kernel_conv_components: evaluate thetaA, A and A2, where A is either the fixed or moving image */
Pitch must be expressed in elements, not bytes!
width (x) corresponds to the fastest dimension (the same as pitch), depth (z) to the slowest
The array-dimension parameters must be the same for all 3 arrays. =#
function kernel_conv_components!(A::T, A2::T, thetaA::T) where T
    Tel = eltype(T)
    i = cucartesianindex(A)
    if checkbounds(Bool, A, i)
        local_A = A[i]
        local_thetaA = isnan(local_A)
        if local_thetaA
            local_A = Tel(0)
            A[i] = local_A
        end
        thetaA[i] = local_thetaA ? Tel(0) : Tel(1)
        A2[i] = local_A * local_A;
    end
    return nothing
end

# CUDA kernel_calcNumDenom: case INTENSITY, compute numerator and denominator before ifftn
function kernel_calcNumDenom_intensity!(f_fft::T, f2_fft::T, thetaf_fft::T,
                                        m_fft::T, m2_fft::T, thetam_fft::T,
                                        numerator_fft::T, denominator_fft::T) where T
    i = cucartesianindex(f_fft)
    if checkbounds(Bool, f_fft, i)
        c1 = f2_fft[i]' * thetam_fft[i]
        c2 = thetaf_fft[i]' * m2_fft[i]
        c1 = c1 + c2
        c2 = f_fft[i]' * m_fft[i]
        c2 = 2 * c2
        numerator_fft[i] = c1 - c2
        denominator_fft[i] = c1
    end
    return nothing
end

# CUDA kernel_calcNumDenom: case PIXELS, compute numerator and denominator before ifftn
function kernel_calcNumDenom_pixels!(f_fft::T, f2_fft::T, thetaf_fft::T,
                                        m_fft::T, m2_fft::T, thetam_fft::T,
                                        numerator_fft::T, denominator_fft::T) where T
    i = cucartesianindex(f_fft)
    if checkbounds(Bool, f_fft, i)
         c1 = -2 * f_fft[i]'
         c1 = c1 * m_fft[i]
         c2 = thetaf_fft[i]' * m2_fft[i]
         c1 = c1 + c2
         c2 = f2_fft[i]' * thetam_fft[i]
         numerator_fft[i] = c1 + c2;
         denominator_fft[i] = thetaf_fft[i]' * thetam_fft[i]
    end
    return nothing
end
