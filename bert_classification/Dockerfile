FROM docker-internal.tcsbank.ru/new_products/default-gpu-pack:cuda-10-1
ARG DATA_DIR='/home/src/data/bpe'
ARG BATCH_SIZE='32'
ARG EPOCHS='10'
ARG CUDA_DEVICES='-1'
ENV DATA_DIR=$DATA_DIR
ENV BATCH_SIZE=$BATCH_SIZE
ENV EPOCHS=$EPOCHS
ENV CUDA_DEVICES=$CUDA_DEVICES
COPY requirements.txt /home/requirements.txt
COPY download_nltk.py /home/download_nltk.py
RUN pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r /home/requirements.txt && rm /home/requirements.txt && \
    pip uninstall -y dataclasses && \
    python /home/download_nltk.py
WORKDIR /home/src/
ENTRYPOINT python train.py
