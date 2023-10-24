# Make sure to be in the scripts directory and activate the environment
# containing mlp-train and other packages required for the chosen machine
# learning potential before running this script.

# We need a newer ASE version to be able to use PLUMED interface
# Here we use a specific commit from the main branch that is known to work.
echo "* Installing ASE package from the master branch *"
python -m pip install git+https://github.com/rosswhitfield/ase@f2615a6e9a
echo "* DONE! *"
