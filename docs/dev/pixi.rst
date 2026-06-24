****************
Using pixi
****************

The MACE environment for *mlp-train* is managed with `pixi <https://pixi.sh>`_.
This page is a short primer; see :doc:`installation` for the install steps.

What pixi is
============

Pixi is a project-level package/environment manager. It reads a ``pixi.toml``
manifest, solves dependencies (conda packages **and** PyPI packages together)
into a ``pixi.lock`` lockfile, and installs everything into a project-local
``.pixi/envs/`` folder. There is no global ``conda activate`` — environments
live with the repository and are reproducible from the lockfile.

Think of it as ``conda`` + ``pip`` + a lockfile + a task runner, scoped to the
project.

One-time setup
==============

.. code-block:: bash

   curl -fsSL https://pixi.sh/install.sh | bash   # installs to ~/.pixi/bin

Restart your shell (or ``source ~/.zshrc``) so ``pixi`` is on ``PATH``.

Core concepts in this repo
==========================

- **Features** are reusable dependency groups (``mace``, ``dev``, ``docs`` in
  ``pixi.toml``).
- **Environments** are named combinations of features. We have two: ``mace``
  (= ``mace`` + ``dev``) and ``docs`` (= ``mace`` + ``docs``).
- **Lockfile** (``pixi.lock``) holds the exact resolved versions. It is
  committed to git and is what makes installs reproducible.

We do not define a ``default`` environment, so **always pass** ``-e mace`` (or
``-e docs``) to pixi commands in this repo.

Day-to-day commands
===================

.. code-block:: bash

   # Create / sync the environment from the lockfile
   pixi install -e mace

   # Run a one-off command inside the env (no activation needed)
   pixi run -e mace python examples/water.py
   pixi run -e mace pytest

   # Run a predefined task (see [tasks] in pixi.toml)
   pixi run -e mace test          # == pytest --cov=mlptrain

   # Drop into an interactive shell with the env activated
   pixi shell -e mace
   # ... work ... then:
   exit

   # List what is installed in an env
   pixi list -e mace

Changing dependencies
=====================

Edit ``pixi.toml`` directly, or use the CLI (which edits it for you):

.. code-block:: bash

   pixi add --feature mace scipy           # conda package onto the mace feature
   pixi add --feature mace --pypi some-pkg  # PyPI package
   pixi remove --feature mace scipy

Either way, re-solve and update the lock, then commit ``pixi.toml`` **and**
``pixi.lock`` together:

.. code-block:: bash

   pixi lock              # re-solve, update pixi.lock only
   pixi install -e mace   # also apply it to the local env

A teammate then just runs ``pixi install -e mace`` to get the identical
environment.

The CUDA gotcha
===============

The lockfile pins **CUDA builds** (``[system-requirements] cuda = "12"`` in
``pixi.toml``). On a machine **without a GPU** (laptop, head node, CI), the
install refuses unless you tell it to pretend a GPU driver is present:

.. code-block:: bash

   CONDA_OVERRIDE_CUDA=12.0 pixi install -e mace

The CUDA-built packages still run fine on CPU — this just lets them install.
(CI and ReadTheDocs set this variable already.)

Reproducibility flags
=====================

- ``pixi install --locked -e mace`` — install strictly from the lockfile and
  **fail** if ``pixi.toml`` and ``pixi.lock`` have drifted (used in CI).
- ``pixi lock --check`` — verify the lockfile is up to date with the manifest
  without changing anything.

Cheat sheet
===========

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Goal
     - Command
   * - Set up the env
     - ``pixi install -e mace``
   * - Run tests
     - ``pixi run -e mace test``
   * - Interactive shell
     - ``pixi shell -e mace``
   * - Run any command
     - ``pixi run -e mace <cmd>``
   * - Add a conda dep
     - ``pixi add --feature mace <pkg>``
   * - Add a PyPI dep
     - ``pixi add --feature mace --pypi <pkg>``
   * - Refresh lockfile
     - ``pixi lock``
   * - Build docs
     - ``pixi run -e docs docs``
   * - Install on CPU-only box
     - ``CONDA_OVERRIDE_CUDA=12.0 pixi install -e mace``
