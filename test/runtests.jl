using Test, ImageCore, ImageFiltering
using CuArrays, CUDAdrv, CUDAnative, RegisterCore, CenterIndexedArrays
import RegisterMismatchCuda
RM = RegisterMismatchCuda

accuracy = 1e-5

devlist = CuDevice[]
map(dev->capability(dev) >= v"2.0" ? push!(devlist,dev) : nothing, devices())
if isempty(devlist)
    error("There is no CUDA device having capability bigger than version 2.0.")
end

@testset "kernel_conv_components" begin
    function run_components(A)
        G1 = CuArray(A)
        G0 = CuArray{eltype(A)}(undef, size(A))
        G2 = CuArray{eltype(A)}(undef, size(A))
        sz = size(A)
        dev = device()
        threadspb = RM.calculate_threads(sz, attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
        nblocks = ceil.(Int, sz ./ threadspb)
        @cuda blocks = nblocks threads = threadspb RM.kernel_conv_components!(G1, G2, G0)
        synchronize()
        A0, A1, A2 = Array(G0), Array(G1), Array(G2)
    end

    for dev in devlist
        device!(dev) do
            try
                A = [1 2; NaN 4]
                B = [NaN NaN; 5 NaN]
                A0, A1, A2 = run_components(A)
                @test A0 == .!isnan.(A)
                @test A1 == [1 2; 0 4]
                @test A2 == [1 4; 0 16]
                B0, B1, B2 = run_components(B)
                @test B0 == .!isnan.(B)
                @test B1 == [0 0; 5 0]
                @test B2 == [0 0; 25 0]
            finally
            end
        end
    end
end

@testset "kernel_calcNumDenom" begin
    function run_cpu(f::Function, f_fft, f2_fft, thetaf_fft, m_fft,m2_fft, thetam_fft)
        numerator_fft = similar(f_fft)
        denominator_fft = similar(f_fft)
        for i in CartesianIndices(f_fft)
            f(i, f_fft, f2_fft, thetaf_fft, m_fft,m2_fft, thetam_fft, numerator_fft, denominator_fft)
        end
        numerator_fft, denominator_fft
    end

    function run_gpu(kernel::Function, f_fft, f2_fft, thetaf_fft, m_fft,m2_fft, thetam_fft)
        d_f = CuArray(f_fft)
        d_f2 = CuArray(f2_fft)
        d_tf = CuArray(thetaf_fft)
        d_m = CuArray(m_fft)
        d_m2 = CuArray(m2_fft)
        d_tm = CuArray(thetam_fft)
        d_numerator = similar(d_f)
        d_denominator = similar(d_f)
        sz = size(f_fft)
        dev = device()
        threadspb = RM.calculate_threads(sz, attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
        nblocks = ceil.(Int, sz ./ threadspb)
        args = (d_f, d_f2, d_tf, d_m, d_m2, d_tm, d_numerator, d_denominator)
        @cuda blocks = nblocks threads = threadspb kernel(args...)
        synchronize()
        Array(d_numerator), Array(d_denominator)
    end

    function calcNumDenom_intensity!(i, f_fft, f2_fft, thetaf_fft, m_fft,m2_fft, thetam_fft, numerator_fft, denominator_fft)
        if checkbounds(Bool, f_fft, i)
            c1 = f2_fft[i]' * thetam_fft[i]
            c2 = thetaf_fft[i]' * m2_fft[i]
            c1 = c1 + c2
            c2 = f_fft[i]' * m_fft[i]
            c2 = 2 * c2
            numerator_fft[i] = c1 - c2
            denominator_fft[i] = c1
        end
    end

    function calcNumDenom_pixels!(i, f_fft, f2_fft, thetaf_fft, m_fft,m2_fft, thetam_fft, numerator_fft, denominator_fft)
        if checkbounds(Bool, f_fft, i)
            c1 = -2 * f_fft[i]'
            c1 = c1 * m_fft[i]
            c2 = thetaf_fft[i]' * m2_fft[i]
            c1 = c1 + c2
            c2 = f2_fft[i]' * thetam_fft[i]
            numerator_fft[i] = c1 + c2;
            denominator_fft[i] = thetaf_fft[i]' * thetam_fft[i]
        end
    end

    for dev in devlist
        device!(dev) do
            A = rand(Complex{Float64},3,3)
            B = rand(Complex{Float64},3,3)
            C = rand(Complex{Float64},3,3)
            D = rand(Complex{Float64},3,3)
            E = rand(Complex{Float64},3,3)
            F = rand(Complex{Float64},3,3)
            num_cpu, denom_cpu = run_cpu(calcNumDenom_intensity!,A,B,C,D,E,F)
            num_gpu, denom_gpu = run_gpu(RM.kernel_calcNumDenom_intensity!,A,B,C,D,E,F)
            @test ≈(num_cpu, num_gpu, atol=accuracy)
            @test ≈(denom_cpu, denom_gpu, atol=accuracy)
            num_cpu, denom_cpu = run_cpu(calcNumDenom_pixels!,A,B,C,D,E,F)
            num_gpu, denom_gpu = run_gpu(RM.kernel_calcNumDenom_pixels!,A,B,C,D,E,F)
            @test ≈(num_cpu, num_gpu, atol=accuracy)
            @test ≈(denom_cpu, denom_gpu, atol=accuracy)
        end
    end
end

@testset "Mismatch 2D test" begin
    for dev in devlist
        device!(dev) do
            A = zeros(5,5)
            A[3,3] = 3
            B = zeros(5,5)
            B[4,5] = 3
            maxshift = (2,3)
            mm = RM.mismatch(A, A, maxshift)
            num, denom = RM.separate(mm)
            RM.truncatenoise!(mm, 0.01)
            @test indmin_mismatch(mm, 0.01) == CartesianIndex((0,0))
            mm = RM.mismatch(A, B, maxshift)
            RM.truncatenoise!(mm, 0.01)
            @test indmin_mismatch(mm, 0.01) == CartesianIndex((1,2))

            # Testing on more complex objects
            # img = rand(map(UInt8,0:255), 256, 256)
            # rng = Any[1:240, 10:250]
            # fixed = map(Float32, img[rng...])
            # moving = map(Float32, img[rng[1].+13, rng[2].-8])
            # maxshift = (20, 20)
            img = rand(map(UInt8,0:255), 60, 60)
            rng = Any[1:40, 10:50]
            fixed = map(Float32, img[rng...])
            moving = map(Float32, img[rng[1].+6, rng[2].-8])
            maxshift = (10, 10)
            mm = RM.mismatch(fixed, moving, maxshift)
            @test indmin_mismatch(mm, 0.01) == CartesianIndex((-6,8))
        end
    end
end

@testset "Mismatch 3D test" begin
    for dev in devlist
        device!(dev) do
            # Test 3d similarly
            Apad = parent(padarray(reshape(1:80*6, 10, 8, 6), Fill(0, (4,3,2))))
            Bpad = parent(padarray(rand(1:80*6, 10, 8, 6), Fill(0, (4,3,2))))
            mm = RM.mismatch(Apad, Bpad, (4,3,2))
            num, denom = RegisterCore.separate(mm)
            mmref = CenterIndexedArray{Float64}(undef, 9, 7, 5)
            for k=-2:2, j = -3:3, i = -4:4
                Bshift = circshift(Bpad,-[i,j,k])
                df = Apad-Bshift
                mmref[i,j,k] = sum(df.^2)
            end
            nrm = sum(Apad.^2)+sum(Bpad.^2)
            @test ≈(mmref.data, num.data, atol=accuracy*nrm)
            @test ≈(fill(nrm,size(denom)), denom.data, atol=accuracy*nrm)
        end
    end
end

@testset "Mismatch apertures 2D test" begin
    for dev in devlist
        device!(dev) do
            for imsz in ((15,16), (14,17))
                for maxshift in ((4,3), (3,2))
                    for gridsize in ((3,3), (2,1), (2,3), (2,2), (1,3))
                        Apad = parent(padarray(reshape(1:prod(imsz), imsz[1], imsz[2]), Fill(0, maxshift, maxshift)))
                        Bpad = parent(padarray(rand(1:20, imsz[1], imsz[2]), Fill(0, maxshift, maxshift)))
                        # intensity normalization
                        mms = RM.mismatch_apertures(Float64, Apad, Bpad, gridsize, maxshift, normalization=:intensity, display=false)
                        nums, denoms = RegisterCore.separate(mms)
                        num = sum(nums)
                        denom = sum(denoms)
                        mm = CenterIndexedArray{Float64}(undef, (2 .* maxshift .+ 1)...)
                        for j = -maxshift[2]:maxshift[2], i = -maxshift[1]:maxshift[1]
                            Bshift = circshift(Bpad,-[i,j])
                            df = Apad-Bshift
                            mm[i,j] = sum(df.^2)
                        end
                        nrm = sum(Apad.^2)+sum(Bpad.^2)
                        @test ≈(mm.data, num.data, atol=accuracy*nrm)
                        @test ≈(fill(nrm,size(denom)), denom.data, atol=accuracy*nrm)
                        # pixel normalization
                        mms = RM.mismatch_apertures(Float64, Apad, Bpad, gridsize, maxshift, normalization=:pixels, display=false)
                        _, denoms = RegisterCore.separate(mms)
                        denom = sum(denoms)
                        n = Vector{Int}[size(Apad,i).-abs.(-maxshift[i]:maxshift[i]) for i = 1:2]
                        @test ≈(denom.data, n[1].*n[2]', atol=accuracy*maximum(denom))
                    end
                end
            end
        end
    end
end

@testset "Mismatch apertures 3D test" begin
    for dev in devlist
        device!(dev) do
            # Test 3d similarly
            Apad = parent(padarray(reshape(1:80*6, 10, 8, 6), Fill(0, (4,3,2))))
            Bpad = parent(padarray(rand(1:80*6, 10, 8, 6), Fill(0, (4,3,2))))
            mms = RM.mismatch_apertures(Apad, Bpad, (2,3,2), (4,3,2), normalization=:intensity, display=false)
            nums, denoms = RegisterCore.separate(mms)
            num = sum(nums)
            denom = sum(denoms)
            mmref = CenterIndexedArray{Float64}(undef, 9, 7, 5)
            for k=-2:2, j = -3:3, i = -4:4
                Bshift = circshift(Bpad,-[i,j,k])
                df = Apad-Bshift
                mmref[i,j,k] = sum(df.^2)
            end
            nrm = sum(Apad.^2)+sum(Bpad.^2)
            @test ≈(mmref.data, num.data, atol=accuracy*nrm)
            @test ≈(fill(sum(Apad.^2)+sum(Bpad.^2),size(denom)), denom.data, atol=accuracy*nrm)
        end
    end
end
