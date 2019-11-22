# # import torch
# # roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
# roberta.eval()

for SPLIT in train dev; do
    python3 -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "meme/data/$SPLIT.input0" \
        --outputs "meme/data/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done
 
fairseq-preprocess \
    --only-source \
    --trainpref "meme/data/train.input0.bpe" \
    --validpref "meme/data/dev.input0.bpe" \
    --destdir "meme/bin/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "meme/data/train.label" \
    --validpref "meme/data/dev.label" \
    --destdir "meme/bin/label" \
    --workers 60





TOTAL_NUM_UPDATES=50  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=3      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=3
MAX_SENTENCES=8       # Batch size.
ROBERTA_PATH=/path/to/roberta/model.pt

python3 train.py meme/bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 16 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 40 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence \
    --find-unused-parameters \
    --update-freq 1