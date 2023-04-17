ARG GCS_ACCESS_KEY
ARG GCS_SECRET_KEY
ARG GCS_BUCKET
ARG MODEL_GCS_PATH="ai_models/FreeVC"
# full MODEL_LST: freevc-24.pth D-freevc-24.pth freevc.pth freevc-s.pth
ARG MODEL_LST="freevc.pth"
ARG MODEL_DIR="/app/checkpoints"
ARG WAVLM_DIR="/app/wavlm"
ARG WAVLM_URL="https://drive.google.com/uc?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU"


FROM python:3.8-slim as model_downloader
ARG GCS_ACCESS_KEY
ARG GCS_SECRET_KEY
ARG GCS_BUCKET
ARG MODEL_GCS_PATH
ARG MODEL_DIR
ARG MODEL_LST
ARG WAVLM_DIR
ARG WAVLM_URL

RUN apt-get update && apt-get install -y wget && \
    pip install gsutil gdown && \
    mkdir -p ${MODEL_DIR} && \
    for m in ${MODEL_LST}; do \
        echo "downloading FreeVC model ${m} ..." && \
        gsutil -m -o "Credentials:gs_access_key_id=${GCS_ACCESS_KEY}" -o "Credentials:gs_secret_access_key=${GCS_SECRET_KEY}" cp gs://${GCS_BUCKET}/${MODEL_GCS_PATH}/${m} ${MODEL_DIR}/${m}; \
    done && \
    echo "modle wavlm ..." && \
    gdown -O ${WAVLM_DIR}/ ${WAVLM_URL}


From nvidia/cuda:11.2.0-base-ubuntu20.04 as runtime

ARG MODEL_DIR
ARG WAVLM_DIR
ENV PATH=/root/miniconda3/bin:$PATH
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app

COPY . .

RUN apt-get update -yq && apt-get install -yq gcc curl wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ./miniconda.sh \
    && ./miniconda.sh -b -p /root/miniconda3 \
    && rm miniconda.sh \
    && conda install -y python==3.8 \
    && conda clean -ya

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY --from=model_downloader ${MODEL_DIR} ${MODEL_DIR}
COPY --from=model_downloader ${WAVLM_DIR} ${WAVLM_DIR}
