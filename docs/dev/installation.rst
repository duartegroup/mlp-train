************
Installation
************

Installation of mlp-train package requires ``conda`` or ``mamba``. If you do not have it already installed, you can download it from https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation.

Mlp-train can be cloned from https://github.com/duartegroup/mlp-train. 

.. code-block:: python

     git clone https://github.com/duartegroup/mlp-train.git

Each machine learning potential (MLP) has its own conda environment and can be installed by executing  corresponding script:


.. code-block:: python

   #Install GAP
   ./install_gap.sh

   #Install ACE
   ./install_ace.sh

   #Install MACE
   ./install_mace.sh


ACE installation requires ``Julia`` (v<=1.6) in the $PATH.

MACE potential benefits from gpu acceleration. To make sure pytorch is installed with CUDA support, you need to either install the environment from machine with GPU access or specify:

.. code-block:: python

   CONDA_OVERRIDE_CUDA="11.8" ./install_mace.sh


The packages are installed into a new conda environments. For MACE, the environment is called mlptrain-mace. To activate it, type:

.. code-block:: python

   conda activate mlptrain-mace

After activation, you can check that the packages are installed with support for CUDA as:

.. code-block:: python

   conda list | grep pytorch

If everything works correctly, you should see something similar to

.. code-block:: python

   pytorch  2.4.1 cuda118_py39ha48351b_305 conda-forge

If the third column does not contain the word `cuda`, you need to install the environment again.
