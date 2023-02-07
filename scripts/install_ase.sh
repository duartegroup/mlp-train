# Make sure you are in the /scripts directory before running this script
echo "Installing ASE package from the master branch"
conda install pip
conda install git
git clone https://github.com/rosswhitfield/ase.git
pip install ./ase
chmod -R +w ./ase && rm -r ./ase
