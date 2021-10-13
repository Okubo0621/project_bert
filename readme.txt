ここ見ろ
https://github.com/google-research/bert

predict_mask.py: 文章の穴あき箇所を，BERTを使って予測する
modelディレクトリ: bertの学習済みモデル
bertディレクトリ: 元々のBERTコード（python2.0）
bert2ディレクトリ: Python3用に書き換えたBERTコード
sample_dataディレクトリ: サンプルの学習データ

使い方：
「ファインチューニング？」
python3 ./bert_v2/extract_features.py --input_file=./input_file/input.txt --output_file=./output_file/output.json --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/model.ckpt-20 --do_lower_case False --layers -1

export BERT_BASE_DIR=‘./model/用いる学習済みファイル名’
-- input_file: 学習データ（1行１文）
-- output_file: 
-- vocab_file: 単語ファイル．このファイルに存在しない単語は不明単語として扱われる．
-- bert_config_file: 設定ファイル？．vocab_sizeの項目だけ変更する
--init_checkpoint: 
--do_lower_case: 
--layers: 必要なtransformerのエンコーダー層で、基本的に-1（-1以降はそんなに変わらないらしい）．複数欲しい時は、-1, -2みたいにカンマ区切り



「事前学習」
１を実行後に、２を実行する。

１：
python3 ./bert_v2/create_pretraining_data.py --input_file=./input_file/sample_text.txt --output_file=./input_file/tf_examples.tfrecord --vocab_file=./model/uncased_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5

--input_file: 学習データ（1行１文）
--output_file: 
--vocab_file: 単語ファイル．このファイルに存在しない単語は不明単語として扱われる．学習データから作る．
--do_lower_case: 
--max_seq_length: 
--max_predictions_per_seq: 
--masked_lm_prob: 
--random_seed:
--dupe_factor: 

2:
python3 ./bert_v2/run_pretraining.py --input_file=./input_file/tf_examples.tfrecord --output_dir=./model/pretraining_output --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --train_batch_size=32 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=20 --num_warmup_steps=10 --learning_rate=2e-5
注意：
・0から学習する場合は、init_chechpointいらない

--input_file: create_pretraining_data.py の出力ファイル
--output_dir: modelディレクトリ下にあるディレクトリと似たようなもの
--do_train: 
--do_eval: 
--bert_config_file:
--init_checkpoint: 0から学習する場合は，いらない
--train_batch_size:
--max_seq_length:
--max_predictions_per_seq:
--num_train_steps:
--num_warmup_steps:
--learning_rate:



