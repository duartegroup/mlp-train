color='\033[0;32m'
no_color='\033[0m'

function print {
    echo -e "$color$1$no_color"
    }

print "Installing QUIP..."
pip install quippy-ase
print "               ...done\n\n"
# -----------------------------------------------------------------------------

print "Installing electronic structure packages.\n
Note: ORCA cannot be installed automatically as the EULA must be accepted individually.
Go to https://orcaforum.kofo.mpg.de/index.php, sign in and go to 'downloads' to download and install."

read -p "Install xtb? ([y]/n)" -r install_xtb

if [ "$install_xtb" == "y" ] || [ "$install_xtb" == "" ]; then
    print "Installing xtb..."
    conda install -c conda-forge xtb --yes
    print "              ...done\n\n"
fi
