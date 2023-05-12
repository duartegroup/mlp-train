# Make sure to be in the scripts directory and activate the environment
# containing mlp-train and other packages required for the chosen machine
# learning potential before running this script.

echo "* Installing ASE package from the master branch *"
wget https://github.com/rosswhitfield/ase/archive/refs/heads/master.zip
unzip master.zip && rm master.zip
pip install ase-master/.
rm -r ase-master

echo "* DONE! *"
