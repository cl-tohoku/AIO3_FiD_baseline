# Fusion-in-Decoder (FiD)

- This directory is forked by [facebookresearch/FiD](https://github.com/facebookresearch/FiD)
- Izacard+'21 - Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (EACL) [[arXiv](https://arxiv.org/abs/2007.01282)] 

## Fusion-in-Decoder (Reader)

Fusion-in-Decoderは、質問と各関連文書を連結したものをエンコーダーでベクトル化し、それを連結したものをデコーダーに入力することで解答を生成するモデルです。

### 環境構築

#### Dockerコンテナの起動
`AIO3_FiD_baseline/generators/fusion_in_decoder`にいる状態で、下記のコマンドに従ってコンテナを起動してください。
```bash
$ docker image build --tag aio3_fid:fid .
$ cd ../../  # "AIO3_FiD_baseline/."に移動してください
$ docker container run \
      --name train_fid \
      --rm \
      --interactive \
      --tty \
      --gpus all \
      --mount type=bind,src=$(pwd),dst=/app \
      aio3_fid:fid \
      bash
```

#### 設定

cuda バージョンに合わせて torch v1.12.0 をインストールして下さい。
```bash
# 実行例（CUDA 11.3 で pip を使用する場合）
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```


### Reader の学習

### データセット

#### 取得

1. 次のコマンドから、[AIO3_FiD_baseline/datasets.yml](../../datasets.yml) ファイルを編集して下さい。

```bash
$ vim datasets.yml
```

このファイルには、Retriever(Dense Passage Retrieval) によって検索された関連文書と質問を含むファイルへのパスを、下記に従って編集して下さい。

```yml
DprRetrieved:
  path: JaqketAIO.load_jaqketaio2
  class: JaqketAIO
  data:
    train: retrievers/AIO3_DPR/${DIR_RESULT}/${exp_name}/retrieved/train_aio_pt.json
    dev: retrievers/AIO3_DPR/${DIR_RESULT}/${exp_name}/retrieved/dev_aio_pt.json
    test: retrievers/AIO3_DPR/${DIR_RESULT}/${exp_name}/retrieved/test_aio_pt.json
```

2. 次に、Fusion-in-Decoder 用にデータセット形式を変換します。

```bash
$ python prepro/convert_dataset.py DprRetrieved fusion_in_decoder
```

### 形式
以下のインスタンスからなる JSONL ファイルを使用します。

```json
{
    "id": "(str) 質問ID",
    "question": "(str) 質問",
    "target": "(str) answers から一つ選択した答え。ない場合はランダムに選択される。",
    "answers": "(List[str]) 答えのリスト",
    "ctxs": [{
        "id": "(int) 記事ID",
        "title": "(str) Wikipedia 記事タイトル",
        "text": "(str) Wikipedia 記事",
        "score": "(float) retriever の検索スコア (ない場合は 1/idx で置換される。generator では使用されない。)",
        "has_answer": "(bool) 'text'内に答えが含まれているかどうか"
    }]
}
```

### 学習

関連文書を用いて Fusion-in-Decoder モデルを学習します。
- [scripts/train_generator.sh](scripts/train_generator.sh)

学習に必要なデータセットのパスを、事前に下記ファイルに設定して下さい。
```bash
$ vim configs/train_generator_slud.yml
```

- [configs/train_generator_slud.yml](configs/train_generator_slud.yml)
    - `name`：訓練の実行名
    - `train_data`：変換後の訓練用データセット（第三回訓練セット）
    - `eval_data`：変換後の開発用データセット（第三回開発セット）
    - `checkpoint_dir`：`name`ディレクトリが配下に作成されるディレクトリ
    - `model_path`：学習したモデルファイルの保存先
    
```bash
# experiment
- name: "fusion-in-decoder"

# model
- train_data: "/app/datasets/fusion_in_decoder/DprRetrieved/train.jsonl"
- eval_data: "/app/datasets/fusion_in_decoder/DprRetrieved/dev.jsonl"
- checkpoint_dir: "<checkpoint_dir>"

# model 
- model_path: "<save_dir>"
```

下記コマンドを実行することで、学習を実行します。

```bash
$ bash scripts/train_generator.sh configs/train_generator_slud.yml
```

### 評価

解答の生成を行います。
評価に必要なデータセットのパスを、事前に下記ファイルに設定して下さい。
- [configs/test_generator_slud.yml](configs/test_generator_slud.yml)
  - `name`：生成される解答テキストファイルの保存先
  - `eval_data`：評価したい変換後のデータセット（第三回開発セット、第三回リーダーボード評価セット）
  - `checkpoint_dir`：`name`ディレクトリが作成されるディレクトリのパス（デフォルト：使用する Reader モデルが保存されているディレクトリ）
  - `model_path`：使用する Reader モデルが保存されているディレクトリへのパス

```bash
$ vim configs/test_generator_slud.yml
```

```bash
# experiment
- name: "fusion-in-decoder_test"

# model
- train_data: "/app/datasets/fusion_in_decoder/DprRetrieved/train.jsonl"
- eval_data: "/app/datasets/fusion_in_decoder/DprRetrieved/test.jsonl"
- checkpoint_dir: "<checkpoint_dir>"

# model 
- model_path: "<save_dir>"
```

下記コマンドを実行することで、解答の生成を行います。
```bash
$ bash scripts/test_generator.sh configs/test_generator_slud.yml
```


## コード

### ディレクトリ構造

```yml
- fid/:
  - data.py:       Dataset/DataLoader
  - evaluation.py: 評価
  - index.py:      FAISS
  - model.py:      Retriever/FiD 定義
  - options.py:    ハイパラ関連（argparse）
  - slurm.py:      GPU 関連（single and multi-GPU / multi-node / SLURM jobs）
  - util.py:       その他 utils
```

### main

```python
# train_generator.py

def train(*):
    model.train()
    while step < args.total_steps:
        for batch in train_dataloader:
            qids, target_ids, target_masks, passage_ids, passage_masks = batch
            outputs = model(
                input_ids = passage_ids.cuda(),
                attention_mask = passage_masks.cuda(),
                labels = target_ids.cuda(),
            )
            train_loss = outputs[0]
            train_loss.backward()

            if step % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
```

### Collater

```python
# fid/data.py

def encode_passages(tokenizer, batch_text_passages, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True
        )
        passage_ids.append(p["input_ids"][None])
        passage_masks.append(p["attention_mask"][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, tokenizer, text_maxlength, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        # batch[0] == Dataset.__getitem__
        assert batch[0]["target"] is not None
        index = torch.tensor([ex["index"] for ex in batch])
        target = [ex["target"] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length = self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length = True,
            return_tensors = "pt",
            truncation = self.answer_maxlength > 0,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            """ encoder に入力する question + passage を作成 """
            if example["passages"] is None:
                return [example["question"]]
            return [example["question"] + " " + t for t in example["passages"]]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(
            self.tokenizer,
            text_passages,
            self.text_maxlength
        )

        return (index, target_ids, target_mask, passage_ids, passage_masks)
```

### FiDT5
- https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/t5#transformers.T5ForConditionalGeneration
- Raffel+'20 - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (JMLR) [[arXiv](https://arxiv.org/abs/1910.10683)]
  - https://aihack.aijobcolle.com/u/Toshiok/85mpjaohigjos5

```python
# fid/model.py

class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=False)
        
    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )


class EncoderWrapper(torch.nn.Module):
    """ Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model. """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        return (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
```
