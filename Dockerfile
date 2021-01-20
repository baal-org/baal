ARG IMAGE_TAG

# ------ Base -----
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04 as base_baal

RUN mkdir -p /app/baal
RUN apt-get update && apt-get install -y --no-install-recommends \
  bzip2 \
  g++ \
  git \
  graphviz \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  wget && \
  rm -rf /var/lib/apt/lists/*

# Install Conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:"$PATH" > /etc/profile.d/conda.sh
RUN conda config --append channels conda-forge
RUN conda install -y h5py
RUN conda install -y pytorch=1.5.0 torchvision cudatoolkit=10.1 -c pytorch

RUN pip install --upgrade pip
COPY requirements.txt /app/baal/requirements.txt
RUN pip install -r /app/baal/requirements.txt
COPY . /app/baal
WORKDIR /app/baal
RUN pip install -e . --no-use-pep517
RUN pip install jupyter cmake
RUN pip install MulticoreTSNE

# ---- test -----
# we need to install test dependencies before, so we cannot use 'base_baal' as base image
FROM continuumio/miniconda3 as test_baal

WORKDIR /app/baal
RUN conda create -n env python=3.7
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

COPY ./test-requirements.txt /app/baal/test-requirements.txt
COPY ./requirements.txt /app/baal/requirements.txt

RUN pip install -r /app/baal/test-requirements.txt
RUN pip install -r /app/baal/requirements.txt
COPY --from=base_baal /app/baal .
RUN pip install -e . --no-use-pep517
