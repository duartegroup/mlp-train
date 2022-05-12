echo "Installing everything to an new conda environment called: ace"
conda create --name ace julia python=3.7 --file ../requirements.txt --yes
conda activate ace
pip install -e ../

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
