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
        "absl-py==1.2.0" \
        "antlr4-python3-runtime==4.8" \
        "asttokens==2.0.5" \
        "backcall==0.2.0" \
        "cachetools==5.2.0" \
        "certifi==2022.6.15" \
        "charset-normalizer==2.1.0" \
        "click==8.1.3" \
        "decorator==5.1.1" \
        "executing==0.8.3" \
        "faiss-gpu==1.7.2" \
        "filelock==3.7.1" \
        "fugashi==1.1.2" \
        "google-auth==2.9.1" \
        "google-auth-oauthlib==0.4.6" \
        "grpcio==1.47.0" \
        "huggingface-hub==0.8.1" \
        "idna==3.3" \
        "importlib-metadata==4.12.0" \
        "ipadic==1.0.0" \
        "ipdb==0.13.9" \
        "ipython==8.4.0" \
        "jedi==0.18.1" \
        "joblib==1.1.0" \
        "markdown==3.4.1" \
        "matplotlib-inline==0.1.3" \
        "mecab-python3==1.0.4" \
        "numpy==1.23.1" \
        "oauthlib==3.2.0" \
        "omegaconf==2.1.1" \
        "packaging==21.3" \
        "pandas==1.3.4" \
        "parso==0.8.3" \
        "pexpect==4.8.0" \
        "pickleshare==0.7.5" \
        "Pillow==9.2.0"  \
        "prompt-toolkit==3.0.30" \
        "protobuf==3.19.0" \
        "ptyprocess==0.7.0" \
        "pure-eval==0.2.2" \
        "pyasn1==0.4.8" \
        "pyasn1-modules==0.2.8" \
        "pygments==2.12.0" \
        "pyparsing==3.0.9" \
        "pyyaml==6.0" \
        "rank_bm25==0.2.1" \
        "regex==2022.7.9" \
        "requests==2.28.1" \
        "requests-oauthlib==1.3.1" \
        "rsa==4.8" \
        "sacremoses==0.0.53" \
        "sentencepiece==0.1.96" \
        "six==1.16.0" \
        "stack-data==0.3.0" \
        "tensorboard==2.8.0" \
        "tensorboard-data-server==0.6.1" \
        "tensorboard-plugin-wit==1.8.1" \
        "tokenizers==0.10.1" \
        "toml==0.10.2" \
        "toolz==0.12.0" \
        "torch==1.9.1" \
        "tqdm==4.62.3" \
        "traitlets==5.3.0" \
        "typing-extensions==4.3.0" \
        "unidic-lite==1.0.8" \
        "urllib3==1.26.10" \
        "wcwidth==0.2.5" \
        "werkzeug==2.1.2" \
        "wheel==0.37.1" \
        "zipp==3.8.1" \
        "cytoolz==0.12.0" \
        "transformers[ja]==4.12.5"

# Download transformers models in advance
ARG TRANSFORMERS_BASE_MODEL_NAME="cl-tohoku/bert-base-japanese-whole-word-masking"
RUN python -c "from transformers import BertModel; BertModel.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"
RUN python -c "from transformers import BertJapaneseTokenizer; BertJapaneseTokenizer.from_pretrained('${TRANSFORMERS_BASE_MODEL_NAME}')"

WORKDIR /code/AIO3_FiD_baseline
