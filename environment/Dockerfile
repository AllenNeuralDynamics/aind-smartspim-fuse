# hash:sha256:3c09b543c48e44c5101c6214c0ea6b12b9be7764a1be670f4e8152f31acd1e1d
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ARG MAVEN_VERSION=3.9.5

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenmpi-dev \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN cd /home && git clone --single-branch --branch fix/data_paths https://github.com/camilolaiton/TeraStitcher.git && \
    mkdir build-terastitcher && \
    cd build-terastitcher && \
    cmake -DWITH_UTILITY_MODULE_teraconverter=ON -DWITH_UTILITY_MODULE_mdatagenerator=ON -DWITH_UTILITY_MODULE_volumeconverter=ON -DWITH_UTILITY_MODULE_mergedisplacements=ON ../TeraStitcher/src && \
    make -j `nproc` && \
    # need ownership of /usr/local to install without sudo
    # chown -R ${USER}:${USER} /usr/local/ && \
    make install && \
    pip install --upgrade autopep8 && \
    autopep8 -i ../TeraStitcher/src/utils/pyscripts/*.py

# ENV PATH "$PATH:/data/TeraStitcher-portable-1.11.10-with-BF-Linux"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/java/openjdk/lib/server"

# Java setup
# OpenJDK setup
RUN cd /home && \ 
    wget https://cdn.azul.com/zulu/bin/zulu8.62.0.19-ca-jdk8.0.332-linux_x64.tar.gz && \
    tar -xzf zulu8.62.0.19-ca-jdk8.0.332-linux_x64.tar.gz
ENV JAVA_HOME "/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/jre/lib/amd64/server"

# Maven setup (for n5-spark utilities)
RUN cd /home && \ 
    wget "https://dlcdn.apache.org/maven/maven-3/$MAVEN_VERSION/binaries/apache-maven-$MAVEN_VERSION-bin.tar.gz" && \
    tar -xzf "apache-maven-$MAVEN_VERSION-bin.tar.gz"
ENV PATH "/home/apache-maven-$MAVEN_VERSION/bin:$PATH"
ENV PATH "/home/zulu8.62.0.19-ca-jdk8.0.332-linux_x64/bin:$PATH"

RUN echo "localhost slots=70" > /home/hostfile

# Creating a new conda environment for argschema python bug
RUN conda create -n smartspim_fusion python=3.8

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "smartspim_fusion", "/bin/bash", "-c"]

RUN pip install -U --no-cache-dir \
    'tifffile < 2022.4.22' \
    xarray_multiscale==1.1.0 \
    argschema==3.0.4 \
    mpi4py \
    bokeh==2.4.2 \
    zarr==2.16.1 \
    pint==0.20.1 \
    aind-data-schema==1.0.0 \
    xmltodict==0.13.0 \
    natsort==8.2.0 \
    pims==0.6.1 \
    dask[distributed]==2022.11.1 \
    aind-ng-link==1.0.6 \
    ome-zarr==0.8.2 \
    psutil==5.9.5 \
    matplotlib==3.7.3 \
    imagecodecs[all]==2023.3.16