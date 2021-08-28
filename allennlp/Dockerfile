FROM docker-internal.tcsbank.ru/new_products/default-gpu-pack:cuda-10-1
ARG DATA_DIR='/home/src/data/'
ARG BATCH_SIZE='32'
ARG EPOCHS='10'
ARG CUDA_DEVICES='-1'
ENV DATA_DIR=$DATA_DIR
ENV BATCH_SIZE=$BATCH_SIZE
ENV EPOCHS=$EPOCHS
ENV CUDA_DEVICES=$CUDA_DEVICES
COPY requirements.txt /home/requirements.txt
RUN pip install -r /home/requirements.txt && rm /home/requirements.txt
WORKDIR /home/src/
ENTRYPOINT allennlp train config/allennlp_config.jsonnet --serialization-dir model --include-package custom || chmod -R 777 .
