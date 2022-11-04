# DPR(Retriever)
BIENCODER_FILE='retrievers/AIO3_DPR/model/baseline/biencoder.pt'
EMBEDDING_FILE='retrievers/AIO3_DPR/model/baseline/embedding.pickle'
PASSAGES_FILE='retrievers/AIO3_DPR/datasets/wiki/jawiki-20220404-c400-large.tsv.gz'
# FiD(Reader)
READER_CONFIG_FILE='generators/fusion_in_decoder/configs/submission_generator_slud.yml'

# Other paths.
RETRIEVER_OUTPUT_FILE='retrievers/AIO3_DPR/result/baseline/submit/test_aio_pt.json'
READER_OUTPUT_FILE='generators/fusion_in_decoder/models_and_results/baseline/fusion-in-decoder_submission/final_output.jsonl'

# Get predictions for all questions in the input.
INPUT_FILE=$1
OUTPUT_FILE=$2


# Now run predictions on input file.
# Retrieve passage from wikipedia.
echo 'Retrieving passages.'
mkdir -p 'retrievers/AIO3_DPR/result/baseline/submit'
python retrievers/AIO3_DPR/dense_retriever.py \
      --model_file $BIENCODER_FILE \
      --ctx_file $PASSAGES_FILE \
      --encoded_ctx_file $EMBEDDING_FILE \
      --qa_file $INPUT_FILE \
      --out_file $RETRIEVER_OUTPUT_FILE \
      --n-docs 100 \
      --validation_workers 32 \
      --batch_size 64 \
      --projection_dim 768

# Convert retrieved results to FiD datasets.
echo 'Converting retrieved results.'
python prepro/convert_dataset.py Submission fusion_in_decoder

# Generate predictions by FiD.
echo 'Reading the retrieved passages.'
python generators/fusion_in_decoder/test_generator.py \
      --config_file $READER_CONFIG_FILE

# Write the predictions to the designed file.
echo 'Writing the prediction.'
cat $READER_OUTPUT_FILE > $OUTPUT_FILE

echo 'Prediction completed.'
