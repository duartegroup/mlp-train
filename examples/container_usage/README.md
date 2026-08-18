# Container Usage 

A dockerfile, and [pre-built
image](docker://ghcr.io/duartegroup/mlp-train:latest), are provided with
mlp-train to provide a funcitoning installation out of the box. Currently the
dockerfile utilises the `environment_mace.yml` to install dependencies into a
micromamba (i.e. conda) environment called `mlptrain-mace`, as per the
`install_mace.sh` installation script, but this is liable to change as the
installation process evolves. 

### Docker

The docker image should be usable as-is, simply pull it from the github
container registry with the command:

``` bash
docker pull ghcr.io/duartegroup/mlp-train:latest
```

and then run the usual way, either interactively: 

``` bash
docker run -it --gpus all --name mlp-train-container ghcr.io/duartegroup/mlp-train:latest /bin/bash
```

or with a specified command by removing `-it` and replacing `/bin/bash` with the
command you'd like to use. Note that you will likely need to mount whatever
specific script you'd like to run onto the container so the container can see
it, see below for an example. Alternatively you could build on top of the docker
image using a `FROM` command in a custom dockerfile, though that is beyond the
scope of this guide. 


### Singularity/Apptainer

If you would like to use the image with singularity you'll need to run a similar
command:

``` bash
singularity build mlp-train.sif docker://ghcr.io/duartegroup/mlp-train:latest
```

which will pull the docker image and convert it into a singularity-friendly OCI
format for you to store locally. Note the preprended `docker://` on the image
address to tell singularity that it's a docker image, not a singularity one.

If you'd then like to run this built image you can use something like:

```bash
singularity exec --nv mlp-train.sif /usr/bin/micromamba run -n mlptrain-mace python /app/examples/water.py
```
where the `water.py` script in this examples directory is run using the
mlptrain-mace conda environment on the container.  

Alternatively you can run the singularity image interactively with:

```bash
singularity shell --nv mlp-train.sif
```

and then run further commands on the container at your leisure. 


## ARC Example Scripts

This folder contains two example scripts for utilising the provided [docker
image](docker://ghcr.io/duartegroup/mlp-train:latest) on an HPC system,
consisting of:

1. [A build script](singularity-build.sh), which builds the singularity image by
   pulling the stored docker image in the github container registry and then
   converting it into a `.sif` file, at some specified location. this is
   submitted as a batch job as ARC's login nodes were _very_ slow to build,
   presumably due to IO constraints. The environment variables
   `SINGULARITY_TMPDIR` and `SINGULARITY_CACHEDIR` are set to local node memory
   and the built `.sif` image is then copied to the user's `$DATA` directory,
   which is ARC-specific but easily portable depending on the particular setup
   of whichever HPC system you are using. 
2. [A run script](singularity-run.sh), which runs the `.sif` image created by
   the build script with singularity's `--nv` flag, which includes the necessary
   nvidia drivers to properly utilise the GPU, as well as `mpirun` to utilise
   however many cores/tasks have been specified. Other notable parts of the
   singularity command include:
    - `-B $DATA/mlp-train:/data`: binds/mounts a local direcory (in this case
       `$DATA/mlp-train`) to a  directory on the container (`/data`), which in
       this case contains the `water_mace.py` script we intend to run.
    - `--pwd /data`: sets the workding directory of the container to this
      mounted directory.
    - `$DATA/singularity/mlp-train.sif`: selects the image built using the above
      build script as the image to be run as a container
    - `/usr/bin/micromamba run -n mlptrain-mace python /data/water_mace.py`:
      runs the intended script (`water_mace.py`) with the micromamba environment
      `mlptrain-mace` on the container 

These are limted examples but should hopefully allow you to start creating your
own scripts for your own purposes. 

Note that the build script need only be re-run if mlp-train is updated.


## Continuous Integration

The docker image is built and pushed to the github container registry upon any
push, or PR, to `main` which changes any of the relevant code for the
dockerfile. Older versions of the docker image, i.e. not latest, can be accessed
with the commit hash of the desired commit. 