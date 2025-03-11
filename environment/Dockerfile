# hash:sha256:3acfa51f7ad1723d1d2b2c3ddf6a0bd8990ab47fd23241c82ff52fdda8eb9754
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV JAVA_HOME=/code/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64/jre
ENV PATH=/code/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64:$PATH
ENV AWS_REGION=us-west-2
ENV JAVA_OPTS=$JAVA_OPTS -Djava.awt.headless=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libblosc-dev=1.21.1+ds2-2 \
        maven=3.6.3-5 \
    && rm -rf /var/lib/apt/lists/*

RUN cd /home && \ 
    wget https://cdn.azul.com/zulu/bin/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz && \
    tar -xzf zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64.tar.gz
ENV JAVA_HOME "/home/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/home/zulu8.80.0.17-ca-fx-jdk8.0.422-linux_x64/jre/lib/amd64/server"

RUN conda install -y \
        ca-certificates==2024.7.4 \
        certifi==2024.7.4 \
        openjdk==8.0.382 \
        openssl==3.3.1 \
    && conda clean -ya

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
    
RUN pip install -U --no-cache-dir aind-data-schema==1.4.0

ENV BIGSTITCHER_HOME=/home/BigStitcher-Spark

COPY postInstall /
RUN chmod +x /postInstall && ./postInstall
