# Install Julia v. 1.6
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.3-linux-x86_64.tar.gz --quiet
tar -xf julia-1.6.3-linux-x86_64.tar.gz
rm -r julia-1.6.3-linux-x86_64.tar.gz

echo "Moving julia install to ~/.julia/"
mkdir ~/.julia && mv julia-1.6.3/* ~/.julia/

echo "Adding ~/.julia/bin to \$PATH"
echo "PATH=$HOME/.julia/bin/:\$PATH" >> ~/.bash_profile
source "$HOME/.bash_profile"

# ----------------------------------------------------
echo "Adding required registries and packages"

echo "using Pkg
Pkg.Registry.add(\"General\")
Pkg.Registry.add(RegistrySpec(url=\"https://github.com/JuliaMolSim/MolSim.git\"))
Pkg.add(PackageSpec(name=\"JuLIP\", version=\"0.10.1\"))
Pkg.add(PackageSpec(name=\"ACE\", version=\"0.8.4\"))
Pkg.add(PackageSpec(name=\"IPFitting\", version=\"0.5.0\"))
Pkg.add(\"IJulia\")
Pkg.add(\"ASE\")" > add.jl
julia add.jl
rm add.jl

# ----------------------------------------------------
echo "Creating a new conda environment for ASE called \"ace\""

conda create --name ace python=3.7 --file ../requirements.txt --yes
conda activate ace

# ----------------------------------------------------
echo "Pointing PyCall to the version of Python in the new env"

echo "ENV[\"PYTHON\"] = \"$(eval "which python")\"
using Pkg
Pkg.build(\"PyCall\")" > pycall.jl
julia pycall.jl
rm pycall.jl

# ---------------------------------------------------
echo "Installing pyjulip"

wget https://github.com/casv2/pyjulip/archive/72280a6ac3.zip
unzip 72280a6ac3.zip && rm 72280a6ac3.zip
cd pyjulip-72280a6ac3f0b4107fe73637fb409b0c42e9011d/ && python setup.py install
cd ..
rm -r pyjulip-72280a6ac3f0b4107fe73637fb409b0c42e9011d/

echo "DONE!"

# Useful env variables:
# JULIA_NUM_THREADS
