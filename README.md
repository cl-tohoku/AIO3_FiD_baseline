![AIO](imgs/aio.png)

- [AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/)

## 更新履歴
- 2022/09/12: 本ベースラインを公開しました。
- 2022/11/05: Dockerイメージの投稿要件に含まれる`~/submission.sh`を追加しました。
- 2022/11/17: Dockerfileに、トークナイザの事前ダウンロードコードを追記しました。また、ベースイメージを変更しました。
- 2022/12/07: DPR および FiD の学習済みモデルの提供を終了しました。従って、ダウンロードスクリプトを実行してもモデルのダウンロードは行えません。


## 目次

- Open-Domain QA
- 初めに
- ディレクトリ構造
- 環境構築
  - コンテナの起動
- [Dense Passage Retrieval](#retriever-dense-passage-retrieval)
    - データセット
      - ダウンロード
    - Retriever
      - 学習済みモデルのダウンロード
      - 設定
      - データセットの質問に関連する文書の抽出
- [Fusion-in-Decoder](#reader-fusion-in-decoder)
    - データセット
      - 作成
      - 形式
    - Reader
      - 学習済みモデルのダウンロード
      - 解答生成と評価
- submission.sh について
- 謝辞・ライセンス



# Open-Domain QA

本実装では、オープンドメイン質問応答に取り組むために二つのモジュール（Retriever-Reader）を使用します。<br>
1. 与えられた質問に対して、文書集合から関連する文書を検索するモジュール（Retriever - Dense Passage Retrieval）
2. 質問と検索した関連文書から、質問の答えを生成するモジュール（Reader - Fusion-in-Decoder）

より詳細な解説は、以下を参照して下さい。

> Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau. Dense Passage Retrieval for Open-Domain Question Answering (EMNLP2020) [\[paper\]](https://www.aclweb.org/anthology/2020.emnlp-main.550) [\[github\]](https://github.com/facebookresearch/DPR)

> Gautier, Izacard and Edouard, Grave. Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering (EACL2021) [\[paper\]](https://aclanthology.org/2021.eacl-main.74.pdf)


## 初めに
リポジトリのクローンとディレクトリの移動を行ってください。
```bash
# コマンド実行例
$ git clone git@github.com:cl-tohoku/AIO3_FiD_baseline.git
$ cd AIO3_FiD_baseline
```


## ディレクトリ構造

```yaml
- datasets.yml:                        データセット定義

# データセットの前処理
- prepro/:
  - convert_dataset.py:                データ形式変換

# Retriever
- retrievers/:
  - AIO3_DPR/:                         DPR モジュール

# 生成型 Reader
- generators/:
  - fusion_in_decoder/:                FiD モジュール
  
- submission.sh:                       システム最終提出用スクリプト
```


## 環境構築
### Dockerコンテナの起動

- まず、Dockerコンテナを起動します。
```bash
# コマンド実行例
$ docker image build --tag aio3_fid:latest .
$ docker container run \
      --name fid_baseline \
      --rm \
      --interactive \
      --tty \
      --gpus all \
      --mount type=bind,src=$(pwd),dst=/app \
      aio3_fid:latest \
      bash
```


## Retriever (Dense Passage Retrieval)

![retriever](imgs/retriever.png)


## データセット

- Retriever (Dense Passage Retrieval) の訓練データには、クイズ大会[「abc/EQIDEN」](http://abc-dive.com/questions/) の過去問題に対して Wikipedia の記事段落の付与を自動で行ったものを使用しています。
- また、開発・評価用クイズ問題には、[株式会社キュービック](http://www.qbik.co.jp/) および [クイズ法人カプリティオ](http://capriccio.tokyo/) へ依頼して作成されたものを使用しています。

- 以上のデータセットの詳細については、[AI王 〜クイズAI日本一決定戦〜](https://www.nlp.ecei.tohoku.ac.jp/projects/aio/) の公式サイト、および下記論文をご覧下さい。

> __JAQKET: クイズを題材にした日本語QAデータセット__
> - https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/
> - 鈴木正敏, 鈴木潤, 松田耕史, ⻄田京介, 井之上直也. JAQKET:クイズを題材にした日本語QAデータセットの構築. 言語処理学会第26回年次大会(NLP2020) [\[PDF\]](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/P2-24.pdf)


### ダウンロード
第三回AI王コンペティションで配布されている訓練・開発・リーダーボード評価用クイズ問題、およびRetriever (Dense Passage Retrieval) の学習で使用するデータセット（訓練・開発用クイズ問題に Wikipedia の記事段落の付与を行ったもの）は、下記のコマンドで取得することができます。
<br>

```bash
$ cd retrievers/AIO3_DPR
$ datasets_dir="datasets"
$ bash scripts/download_data.sh $datasets_dir
```

```bash
# ダウンロードされたデータセット
<datasets_dir>
|- aio_02_train.jsonl              # 第三回訓練データ
|- aio_02_dev_v1.0.jsonl           # 第三回開発データ
|- aio_03_test_unlabeled.jsonl     # 第三回リーダーボード評価データ
|- wiki/
|  |- jawiki-20220404-c400-large.tsv.gz  # Wikipedia 文書集合
|- retriever/
|  |- aio_02_train.json.gz         # 第三回訓練データに Wikipedia の記事段落の付与を行ったもの
|  |- aio_02_dev.json.gz           # 第三回開発データに Wikipedia の記事段落の付与を行ったもの

# 「質問」と「正解」からなる TSV 形式のファイル
|  |- aio_02_train.tsv
|  |- aio_02_dev.tsv
```

| データ             |ファイル名|質問数|       文書数 |
|:----------------|:---|---:|----------:|
| 訓練              |aio\_02\_train|22,335|         - |
| 開発              |aio\_02\_dev\_v1.0|1,000|         - |
| リーダーボード投稿用評価データ |aio\_03\_test\_unlabeled|1,000|         - |
|文書集合|jawiki-20220404-c400-large|-| 4,288,199 |

- データセットの構築方法の詳細については、[retrievers/AIO3_DPR/data/README.md](retrievers/AIO3_DPR/data/README.md)を参照して下さい。


### 学習済みモデルのダウンロード
- 本節では既に学習・作成された、Retriever・文書エンベッディングのダウンロード方法について説明します。
必要に応じてダウンロードし、解凍して下さい。
- なお、Retriever の学習、および文書集合（Wikipedia）のエンコード方法の詳細については、[retrievers/AIO3_DPR/README.md](retrievers/AIO3_DPR/README.md)を参照して下さい。

（注意事項：学習済みモデルの提供は 2022/12/07 をもって終了いたしました。そのため、こちらのスクリプトを実行してもモデルのダウンロードは行えません。）

```bash
$ save_dir="model/baseline"
$ targets="retriever,embeddings"  # {retriever, embeddings} からダウンロード対象を「スペースなしの ',' 区切り」で指定して下さい

$ bash scripts/download_model.sh $targets $save_dir
$ gunzip ${save_dir}/*.gz
$ du -h ${save_dir}/*
  2.5G    biencoder.pt
  13G     embedding.pickle
```

### 設定

```bash
$ vim scripts/configs/config.pth
```

- データセットやモデルを任意の場所に保存した方は、上記設定ファイルに以下の項目を設定して下さい。
    - `WIKI_FILE`：Wikipedia の文書集合ファイル
    - `TRAIN_FILE`：第三回訓練データ
    - `DEV_FILE`：第三回開発データ
    - `TEST_FILE`：第三回リーダーボード評価データ
    - `DIR_DPR`：モデルや文書エンベッディングが保存されているディレクトリへのパス
    - `DIR_RESULT`: 関連文書抽出結果の保存先


### データセットの質問に関連する文書の抽出
データセットの質問に関連する文書を抽出します。質問エンコーダから取得した質問エンベッディングと文書エンベッディングに対して Faiss を用いて類似度を計算します。
- [retrievers/AIO3_DPR/scripts/retriever/retrieve_passage.sh](retrievers/AIO3_DPR/scripts/retriever/retrieve_passage.sh)

```bash
# 実行例

$ exp_name="baseline"
$ model="${save_dir}/biencoder.pt"
$ embed="${save_dir}/embedding.pickle"

$ bash scripts/retriever/retrieve_passage.sh \
    -n $exp_name \
    -m $model \
    -e $embed

# 実行結果
$ ls ${DIR_RESULT}/${exp_name}/retrieved
    train_aio_pt.json   dev_aio_pt.json   test_aio_pt.json   # 予測結果（reader 学習データ）
    train_aio_pt.tsv    dev_aio_pt.tsv    test_aio_pt.tsv    # 予測スコア（Acc@k を含む）
    logs/
      predict_aio_pt.log                               # 実行時ログ
```

__Acc@k__
- 抽出した上位 k 件までの文書に解答が含まれている質問数の割合
- 実行結果のtsvファイルをご参照ください

| データ      | Acc@1 | Acc@5 | Acc@10 | Acc@50 | Acc@100 |
|:---------|------:|------:|-------:|-------:|--------:|
| 第三回訓練セット | 51.76 | 76.07 |  81.93 |  89.25 |   90.89 |
| 第三回開発セット    |  42.2 |  67.8 |   73.3 |      85 |      88 |

<br>

## Reader (Fusion-in-Decoder)

Fusion-in-Decoder(FiD) は、質問と各関連文書を連結したものをエンコーダーでベクトル化し、それを連結したものをデコーダーに入力することで解答を生成するモデルです。


## データセット

### 作成

前節のRetrieverによる関連文書抽出結果を任意の場所に保存した方は、[/app/datasets.yml](datasets.yml) ファイルを編集して下さい。

```bash
$ cd /app
$ vim datasets.yml
```

このファイルには、Retriever(Dense Passage Retrieval) によって検索された関連文書と質問を含むファイルへのパスを、下記に合わせて設定して下さい。

```yml
DprRetrieved:
  path: JaqketAIO.load_jaqketaio2
  class: JaqketAIO
  data:
    train: retrievers/AIO3_DPR/${DIR_RESULT}/${exp_name}/retrieved/train_aio_pt.json
    dev: retrievers/AIO3_DPR/${DIR_RESULT}/${exp_name}/retrieved/dev_aio_pt.json
    test: retrievers/AIO3_DPR/${DIR_RESULT}/${exp_name}/retrieved/test_aio_pt.json
```

<hr>

設定が完了したら、次に Reader 用にデータセット形式を変換します。

```bash
$ python prepro/convert_dataset.py DprRetrieved fusion_in_decoder
```

変換後のデータセットは次のディレクトリに保存されます。

```yaml
/app/datasets/fusion_in_decoder/DprRetrieved/*.jsonl
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

リーダーボード投稿用評価データでは答えが含まれていないため、次のインスタンスからなる JSONL ファイルを使用します。

```json
{
    "id": "(str) 質問ID",
    "question": "(str) 質問",
    "target": "(str) 空文字列",
    "ctxs": [{
        "id": "(int) 記事ID",
        "title": "(str) Wikipedia 記事タイトル",
        "text": "(str) Wikipedia 記事",
        "score": "(float) retriever の検索スコア (ない場合は 1/idx で置換される。generator では使用されない。)",
        "has_answer": "(bool) 'text'内に答えが含まれているかどうか"
    }]
}
```

## Reader による解答生成と評価
はじめに、下記ディレクトリに移動して下さい。
```bash
$ cd generators/fusion_in_decoder
```

### 学習済みモデルのダウンロード
- 本節では既に学習された Reader のダウンロード方法について説明します。
必要に応じてダウンロードし、解凍して下さい。<br>
- また、Reader (Fusion-in-Decoder) の学習については、[generators/fusion_in_decoder/README.md](generators/fusion_in_decoder/README.md)を参照して下さい。

（注意事項：学習済みモデルの提供は 2022/12/07 をもって終了いたしました。そのため、こちらのスクリプトを実行してもモデルのダウンロードは行えません。）

```bash
$ fid_save_dir="models_and_results/baseline"
$ targets="reader"

$ bash scripts/download_model.sh $targets $fid_save_dir
$ gunzip ${fid_save_dir}/*.gz
$ du -h ${fid_save_dir}/*
  4.0K       config.json
  1.7G       optimizer.pth.tar
  851M       pytorch_model.bin
```


### 解答生成と評価

#### 設定

```bash
$ vim configs/test_generator_slud.yml
```

- データセットなどを任意の場所に保存した方は、上記設定ファイルに以下の項目を設定して下さい。
    - `name`：生成される解答テキストファイルの保存先
    - `eval_data`：評価したい変換後のデータセットへのパス（第三回開発セット、第三回リーダーボード評価セット）
    - `checkpoint_dir`：`name`ディレクトリが作成されるディレクトリのパス（デフォルト：使用する Reader モデルが保存されているディレクトリ）
    - `model_path`：使用する Reader モデルが保存されているディレクトリへのパス

#### 解答生成

学習済み生成モデルにより、解答を生成します。<br>
下記スクリプトを実行することで、質問に対する答えをReaderが生成します。
- [scripts/test_generator.sh](generators/fusion_in_decoder/scripts/test_generator.sh)

```bash
# 実行例
$ bash scripts/test_generator.sh configs/test_generator_slud.yml

# 実行結果
$ ls ${checkpoint_dir}/${name}
    final_output.jsonl         # 生成された解答が出力されたファイル
```

#### 評価

__Accuracy__
- 関連文書の上位 60 件の文書を用いた時の正解率 (Exact Match)

| データ      |  Acc |
|:---------|-----:|
| 第三回開発セット | 55.9 |


※ 評価の際に出力されたログを参照して下さい：

```bash
# 出力例
202y-mm-dd hh:mm:ss #106 INFO __main__ :::  EM 0.559000, Total number of example 1000
```

- 関連文書の上位 60 件の文書を用いた時の、リーダーボード投稿用評価データに対する解答出力の例
```json lines
{"qid": "AIO02-1001", "prediction": "セブンマイル橋"}
{"qid": "AIO02-1002", "prediction": "チンダル現象"}
{"qid": "AIO02-1003", "prediction": "ショーシャンクの空に"}
{"qid": "AIO02-1004", "prediction": "ダブルデシジョン"}
{"qid": "AIO02-1005", "prediction": "デュース"}
```



## submission.sh について
最終的に提出を行う Docker イメージ内で、与えられた質問データに対して推論を行うスクリプトです。

このスクリプトを実行する際には、`Dockerfile`に下記のコードを記載した上で Docker イメージのビルドを行ってください。
```bash
ENV TRANSFORMERS_OFFLINE=1
```

その他、提出する Docker イメージの要件や、このスクリプトの実行方法については、[システム最終投稿の方法](https://sites.google.com/view/project-aio/competition3/how-to-submit) をご参照ください。

`submission.sh`は、以下の内容で構成されています：
- `BIENCODER_FILE`：Retriever(DPR) モデルファイルのパス
- `EMBEDDING_FILE`：文書エンベッディングファイルのパス
- `PASSAGES_FILE`：文書集合のパス
- `READER_CONFIG_FILE`：Reader(FiD) モデルを動かすための設定ファイルのパス
- `RETRIEVER_OUTPUT_FILE`：Retriever(DPR) モデルによる文書抽出結果ファイルの出力先
- `READER_OUTPUT_FILE`：Reader(FiD) モデルが出力した解答ファイルへのパス。[解答生成](#解答生成)の節を参照してください。

Reader(FiD) モデルファイルのパスを指定する必要がある場合は、`READER_CONFIG_FILE`内の`model_path`にて指定してください。



## 謝辞・ライセンス

- 学習データに含まれるクイズ問題の著作権は [abc/EQIDEN 実行委員会](http://abc-dive.com/questions/) に帰属します。東北大学において研究目的での再配布許諾を得ています。
- 開発データは クリエイティブ・コモンズ 表示 - 継承 4.0 国際 ライセンスの下に提供されています。
  - <img src="https://i.imgur.com/7HLJWMM.png" alt="" title="">
- 開発/評価用クイズ問題は [株式会社キュービック](http://www.qbik.co.jp/) および [クイズ法人カプリティオ](http://capriccio.tokyo/) へ依頼して作成されたものを使用しております。
