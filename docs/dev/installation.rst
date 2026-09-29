************
Installation
************

Mlp-train can be cloned from https://github.com/duartegroup/mlp-train.

.. code-block:: bash

     git clone https://github.com/duartegroup/mlp-train.git

MACE can be installed either with ``conda`` or with `pixi <https://pixi.sh>`_.

MACE (conda)
============

This route requires ``conda`` or ``mamba``. If you do not have it already
installed, you can download it from
https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation.

From the repository root:

.. code-block:: bash

   ./install_mace.sh

MACE benefits from GPU acceleration. To make sure pytorch is installed with CUDA
support, either install from a machine with GPU access, or override the detected
CUDA version (typical when installing from a head node without GPUs but intending
to run on GPUs):

.. code-block:: bash

   CONDA_OVERRIDE_CUDA=12.0 ./install_mace.sh

The packages are installed into a new conda environment called ``mlptrain-mace``.
To activate it and check that pytorch has CUDA support:

.. code-block:: bash

   conda activate mlptrain-mace
   conda list | grep pytorch

If everything works correctly, you should see something similar to

.. code-block:: text

   pytorch  2.4.1 cuda118_py39ha48351b_305 conda-forge

If the third column does not contain the word ``cuda``, you need to install the
environment again.

MACE (pixi)
===========

The MACE environment can also be managed with `pixi <https://pixi.sh>`_. First install pixi:

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | bash

Then, from the repository root, create the environment (this also installs
``mlptrain`` in editable mode) and run the tests:

.. code-block:: bash

   pixi install -e mace
   pixi run -e mace test

The pixi environment currently supports ``osx-arm44`` and ``linux-64``, either CPU or CUDA-enabled builds (matching
``[system-requirements] cuda = "12"`` in ``pixi.toml``). CUDA build install also `cuEquivariance <https://docs.nvidia.com/cuda/cuequivariance/>`_.
To install CUDA version on a machine without a
GPU (e.g. a head node, or to install CUDA builds for later GPU use), set the CUDA
override so the locked CUDA packages can be installed:

.. code-block:: bash

   CONDA_OVERRIDE_CUDA=12.0 pixi install -e mace

You can open a shell inside the environment with ``pixi shell -e mace`` and check
that pytorch is installed with CUDA support:

.. code-block:: bash

   pixi run -e mace python -c "import torch; print(torch.__version__)"

ACE (conda)
===========

ACE is still installed into its own conda environment via the install script,
which requires ``conda`` or ``mamba`` and ``Julia`` (v<=1.6) in the ``$PATH``:

.. code-block:: bash

   ./install_ace.sh
