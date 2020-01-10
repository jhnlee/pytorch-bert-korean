FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN pip install pytorch_transformers flask pandas

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

WORKDIR /data
