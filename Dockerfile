FROM mambaorg/micromamba:2.1.1-cuda12.9.0-ubuntu22.04

USER root
WORKDIR /app

ENV CONDA_ENV_NAME=mlptrain-mace
ENV CONDA_ENV_FILE=environment_mace.yml

# Install dependencies and MACE
COPY ./${CONDA_ENV_FILE} /app/${CONDA_ENV_FILE}
RUN micromamba env create -n "${CONDA_ENV_NAME}" --file /app/${CONDA_ENV_FILE} && \
    micromamba clean -a

# Install mlptrain
COPY . /app

RUN micromamba run -n ${CONDA_ENV_NAME} python -m pip install -e . && \
    micromamba clean -a

# Had to hardcode the environment name due to docker limitations
ENTRYPOINT ["/usr/bin/micromamba", "run", "-n", "mlptrain-mace"]
CMD ["python", "-m", "pytest"]
