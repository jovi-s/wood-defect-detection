#########################################################
## Python Environment with CUDA
#########################################################

FROM nvidia/cuda:11.3.0-devel-ubuntu20.04 AS python_base_cuda
LABEL MAINTAINER="Jovinder Singh: https://github.com/jovi-s/"

# Update system and install wget
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y wget ffmpeg libpython3.8 git sudo

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh --quiet && \
    bash ~/miniconda.sh -b -p /opt/conda
ENV PATH "/opt/conda/bin:${PATH}" && conda install python=3.8

# Install all anomalib requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# Run Dashboard
CMD ["python", "main.py"]
