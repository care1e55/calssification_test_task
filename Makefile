build_allennlp:
	nvidia-docker build 'allennlp/junior_classification' -t junior_classification

build_bert_classification:
	nvidia-docker build bert_classification -t bert_classification

train_bert_classification:
	python3 bert_classification/train.py

train_bert_classification_slurm:
	nvidia-docker run --name bert_classification -v $(pwd)/bert_classification:/home/src bert_classification

predict_bert:
	python3 bert_classification/predict.py \
		--input_path 'bert_classification/test.txt' \
		--output_path 'bert_classification/bert_preds.csv' \
		--model_path 'bert_classification/bert_for_classification.model' 

train_allennlp:
	nvidia-docker run --name junior_classification \
		--env DATA_DIR="/home/src/data" \
		--env BATCH_SIZE=32 \
		--env EPOCHS=10 \
		--env CUDA_DEVICES=2 \
	    -v $(pwd)/allennlp:/home/src junior_classification

train_fastext:
	python3 fasttext/train.py \
		--train_path 'fasttext/data/train.txt' \
		--valid_path 'fasttext/data/valid.txt' \
		--output_dir 'fasttext/data' \
		--n_trials 50

predict_fasttext:
	python3 fasttext/predict.py

fasttext_dataset:
	python3 make_dataset.py \
		--input_dataset 'junior_no_test.csv' \
		--output_dir 'train_fasstext\data' \
		--type fasstext \

allennlp_dataset:
	python make_dataset.py \
		--input_dataset  'junior_no_test.csv' \
		--output_dir 'train_fasstext\data' \
		--type allennlp \
