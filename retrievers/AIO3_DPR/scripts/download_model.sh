#!/bin/bash
USAGE="bash $0 [targets=retriever,embeddings] [output_dir]"
DATE=`date +%Y%m%d-%H%M`

set -e

TARGETS=($(echo $1 | tr ',' "\n"))
DEST=$2

if [ -z $DEST ] ; then
  echo "[ERROR] Please specify the 'output_dir'"
  echo $USAGE
  exit 1
elif [ ! -z $3 ] ; then
  echo "[ERROR] The arguemnt of $0 takes 2 positional arguments but 3 (or more) was given"
  echo $USAGE
  exit 1
fi


for target in ${TARGETS[@]} ; do
  if [ $target = "retriever" ] ; then
    echo "- Download: retriever"
    wget -nc https://aio-bucket.s3.ap-northeast-1.amazonaws.com/data/aio_03/dpr_model/biencoder.pt.gz -P $DEST
  elif [ $target = "embeddings" ] ; then
    echo "- Download: embeddings"
    wget -nc https://aio-bucket.s3.ap-northeast-1.amazonaws.com/data/aio_03/dpr_model/embedding.pickle.gz -P $DEST
  else
    echo "[WARNING] '${target}' is not appropriate"
    echo "[WARNING] Please specify the target from {retriever, embeddings}"
  fi
done
