# hash:sha256:0784d5498688d2726b285ed04715dbde94b3331709c6e578601baf5ecb42fd99
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    xarray_multiscale==2.1.0 \
    ome-zarr==0.9.0

COPY postInstall /
RUN /postInstall
