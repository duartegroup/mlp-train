# The installation only works for Linux systems
# Make sure to be in the scripts directory before running this script
echo "* Installing everything to an new conda environment called: mace *"

# ----------------------------------------------------
echo "* Installing mlp-train dependencies *"
conda create --name mace python=3.7 --file ../requirements.txt -c conda-forge --yes

conda activate mace

# ----------------------------------------------------
echo "* Installing mlp-train *"
pip install -e ../

# ----------------------------------------------------
echo "* Installing PyTorch *"
pip3 install torch torchvision torchaudio

# ----------------------------------------------------
echo "* Installing MACE dependencies *"
pip install --upgrade e3nn
pip install torch-ema

# ----------------------------------------------------
echo "* Installing MACE *"
wget https://github.com/ACEsuit/mace/archive/refs/tags/v0.2.0.zip
unzip v0.2.0.zip && rm v0.2.0.zip
pip install mace-0.2.0/.
rm -r mace-0.2.0

echo "* DONE! *"
