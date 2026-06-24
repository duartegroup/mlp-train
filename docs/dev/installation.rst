************
Installation
************

Mlp-train can be cloned from https://github.com/duartegroup/mlp-train.

.. code-block:: bash

     git clone https://github.com/duartegroup/mlp-train.git

MACE (pixi)
===========

The MACE environment is managed with `pixi <https://pixi.sh>`_. First install pixi:

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | bash

Then, from the repository root, create the environment (this also installs
``mlptrain`` in editable mode) and run the tests:

.. code-block:: bash

   pixi install -e mace
   pixi run -e mace test

The pixi environment targets ``linux-64`` and ships CUDA-enabled builds (matching
``[system-requirements] cuda = "12"`` in ``pixi.toml``). On a machine without a
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
