# modified from Zack Li

using Pixell, WCS, XGPaint
using Cosmology
using Interpolations
import XGPaint: AbstractProfile
using HDF5
import JSON
using JLD2
using ThreadsX
using FileIO

print("Threads: ", Threads.nthreads(), "\n")

# box = [30   -30;           # RA
#        -25     25] * Pixell.degree  # DEC
# shape, wcs = geometry(CarClenshawCurtis{Float64}, box, 0.5 * Pixell.arcminute)
shape, wcs = fullsky_geometry(0.5 * Pixell.arcminute)

##
fid = h5open("/mnt/raid-cita/mlokken/buzzard/catalogs/halos/buzzard_halos_zgtrpt03.hdf5", "r")
# fid = h5open("/mnt/scratch-lustre/mlokken/pkpatch/halos_hdf5.h5", "r")
ra, dec = deg2rad.(fid["ra"]), deg2rad.(fid["dec"])
redshift = collect(fid["z"])
halo_mass = collect(fid["m200c"])
# choose the cutoff for the pressure profiles here
cutoff = 4
##
perm = sortperm(dec, alg=ThreadsX.MergeSort)
ra = ra[perm]
dec = dec[perm]
redshift = redshift[perm]
halo_mass = halo_mass[perm]

# precomputed sky angles
α_map, δ_map = posmap(shape, wcs)
psa = (sin_α=sin.(α_map), cos_α=cos.(α_map), sin_δ=sin.(δ_map), cos_δ=cos.(δ_map))

##
print("Precomputing the model profile grid.\n")

# set up a profile to paint
p = XGPaint.BattagliaProfile(Omega_c=0.24, Omega_b=0.046, h=0.7) # Buzzard cosmology

# beam stuff (not used in this particular script)
N_logθ = 512
rft = RadialFourierTransform(n=N_logθ, pad=256)


model_file::String = "cached_battaglia.jld2"
if isfile(model_file)
    print("Found cached Battaglia profile model. Loading from disk.\n")
    model = load(model_file)
    prof_logθs, prof_redshift, prof_logMs, prof_y = model["prof_logθs"], 
        model["prof_redshift"], model["prof_logMs"], model["prof_y"]
else
    print("Didn't find a cached profile model. Computing and saving.\n")
    logθ_min, logθ_max = log(minimum(rft.r)), log(maximum(rft.r))
    @time prof_logθs, prof_redshift, prof_logMs, prof_y = profile_grid(p; 
        N_logθ=N_logθ, logθ_min=logθ_min, logθ_max=logθ_max)
    save(model_file, Dict("prof_logθs"=>prof_logθs, 
        "prof_redshift"=>prof_redshift, "prof_logMs"=>prof_logMs, "prof_y"=>prof_y))
end


itp = Interpolations.interpolate(log.(prof_y), BSpline(Cubic(Line(OnGrid()))))
sitp = scale(itp, prof_logθs, prof_redshift, prof_logMs);

##
function paint_map!(m, p::XGPaint.AbstractProfile, psa, sitp, masses, 
                    redshifts, αs, δs, irange; mult=4)
    for i in irange
        α₀ = αs[i]
        δ₀ = δs[i]
        mh = masses[i]
        z = redshifts[i]
        θmax = XGPaint.θmax(p, mh * XGPaint.M_sun, z, mult=mult)
        profile_paint!(m, α₀, δ₀, psa, sitp, z, mh, θmax)
    end
end

function chunked_paint!(m, p::XGPaint.AbstractProfile, psa, sitp, masses, 
                        redshifts, αs, δs; mult=4)
    m .= 0.0
    
    N_sources = length(masses)
    chunksize = ceil(Int, N_sources / (2Threads.nthreads()))
    chunks = chunk(N_sources, chunksize);
    
    Threads.@threads for i in 1:Threads.nthreads()
        chunk_i = 2i
        i1, i2 = chunks[chunk_i]
        paint_map!(m, p, psa, sitp, masses, redshifts, αs, δs, i1:i2, mult=mult)
    end

    Threads.@threads for i in 1:Threads.nthreads()
        chunk_i = 2i - 1
        i1, i2 = chunks[chunk_i]
        paint_map!(m, p, psa, sitp, masses, redshifts, αs, δs, i1:i2, mult=mult)
    end
end

##
# cut = eachindex(halo_mass)[begin:end-5]
m = Enmap(zeros(shape), wcs)

print("Painting map.\n")
@time chunked_paint!(m, p, psa, sitp, halo_mass, redshift, ra, dec, mult=cutoff)


#
write_map(
    "/mnt/raid-cita/mlokken/buzzard/ymaps/ymap_buzzard_standard_bbps_car_zgtrpt03.fits",
    m)

#
using PyPlot
plt.clf()
plt.figure()
plt.imshow(log10.(m.data'))
plt.axis("off")
plt.savefig("test_buzzard.png", bbox_inches="tight",pad_inches = 0)
plt.gcf()



# ##
# m .= 0
# profile_paint!(m, 0., 0., psa, sitp, 0.1, 1e14, π/90)

# ##
# using Plots
# Plots.plot(log10.(m))
