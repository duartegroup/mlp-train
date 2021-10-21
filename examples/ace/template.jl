using IPFitting, ACE, JuLIP, LinearAlgebra
using JuLIP.MLIPs: combine, SumIP
using ACE: z2i, i2z, order
BLAS.set_num_threads(4) # number of threads for the LSQ solver


# first define the ACE basis specification
# specify the species
species = [:H, :C]
N = 4 # maximum correlation order

zC = AtomicNumber(:C)
zH = AtomicNumber(:H)

# maximum degrees for each correlation order
Dd = Dict("default" => 10,
             1 => 10,
             2 => 8,
             (3, zC) => 8,
             (3, zH) => 0,
             (4, zC) => 7,
             (4, zH) => 0)

# for the basis function specified by (n, l)
# degree = n_weight * n + l_weight * l
#n_weights
Dn = Dict( "default" => 1.0 )
#l_weights
Dl = Dict( "default" => 1.5 )

# r0 is a typical length scale for the distance transform
r0 = 1.3
r_in = 0.95 # inner cutoff of ACE, choose a little more than min dist in dataset
r_cut = 4.0 # outer cutoff of ACE

# Specify the pair potential
deg_pair = 5
r_cut_pair = 5.0

# specify the one-body reference potential

E0_C = -1027.5209768231446
E0_H = -13.61362435311999
Vref = OneBody(:H => E0_H, :C => E0_C);

# load the training data
train_data = IPFitting.Data.read_xyz("active_gap.xyz", energy_key="dft_energy",
                                   force_key="dft_forces", virial_key="dummy");
# give weights for the different config_type-s
weights = Dict(
        "default" => Dict("E" => 20.0, "F" => 1.0 , "V" => 0.0 )
        );

dbname = "" # change this from empty to something if you want to save the design matrix

# specify the least squares solver, there are many implemented in IPFitting,
# here are two examples with sensible defaults

# Iterative LSQR with Laplacian scaling
damp = 0.1 # weight in front of ridge penalty, range 0.5 - 0.01
rscal = 2.0 # power of Laplacian scaling of basis functions,  range is 1-4
solver = (:itlsq, (damp, rscal, 1e-6, identity))

# simple riddge regression
# r = 1.05 # how much of the training error to sacrifise for regularisation
# solver = (:rid, r)

save_name = "ACE_DielsAlder.json"

####################################################################################################

Deg = ACE.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

# construction of a basic basis for site energies
Bsite = rpi_basis(species = species,
                   N = N,
                   r0 = r0,
                   D = Deg,
                   rin = r_in, rcut = r_cut,   # domain for radial basis (cf documentation)
                   maxdeg = 1.0, #maxdeg increases the entire basis size;
                   pin = 2)     # require smooth inner cutoff

# pair potential basis
Bpair = pair_basis(species = species, r0 = r0, maxdeg = deg_pair,
                   rcut = r_cut_pair, rin = 0.0,
                   pin = 0 )   # pin = 0 means no inner cutoff


B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

println("The total number of basis functions is")
@show length(B)

dB = LsqDB(dbname, B, train_data);


IP, lsqinfo = IPFitting.Lsq.lsqfit(dB, Vref=Vref,
             solver=solver,
             asmerrs=true, weights=weights)
save_dict(save_name,
           Dict("IP" => write_dict(IP), "info" => lsqinfo))
rmse_table(lsqinfo["errors"])
println("The L2 norm of the fit is ", round(norm(lsqinfo["c"]), digits=2))