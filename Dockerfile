FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ARG wandb_key

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workspace

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install git nano ffmpeg libsm6 libxext6 uuid-runtime
RUN conda install --rev 1
RUN conda install python=3.8
RUN conda install ipykernel ipywidgets pytorch torchvision cudatoolkit=11.3 -c pytorch

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

RUN if [[ "$wandb_key" = "" ]] ; then echo WandB API key not provided ; else wandb login $wandb_key; fi