#!/bin/bash

TRAIN_DIR=../../data/unigram-corpus
VALID_DIR=../../data/unigram-corpus
TEST_DIR=../../data/unigram-corpus

fairseq-preprocess \
    --destdir ../../data/clean250-bin \
    --source-lang en \
    --target-lang ja \
    --trainpref $TRAIN_DIR/clean250 \
    --validpref $VALID_DIR/valid \
    --testpref $TEST_DIR/test \
    --workers `nproc`