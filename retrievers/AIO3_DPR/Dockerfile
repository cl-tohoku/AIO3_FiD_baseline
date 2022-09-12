FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        jq \
        wget \
        vim \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install requirements
RUN pip install --no-cache-dir \
        "mecab-python3==1.0.4" \
        "pandas==1.3.4" \
        "tqdm==4.62.3" \
        "rank_bm25==0.2.1" \
        "transformers[ja]==4.12.5" \
        "faiss-cpu==1.7.1.post2" \
        "tensorboard==2.7.0" \
        "torch==1.9.1" \
        "Pillow==9.2.0"

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-whole-word-masking"
RUN python -c "from transformers import BertModel; BertModel.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN python -c "from transformers import BertJapaneseTokenizer; BertJapaneseTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /code/AIO3_FiD_baseline/retrievers/AIO3_DPR
