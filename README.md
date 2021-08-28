## Classification test task

Репозиторий с решением тестовой задачи классификации

Приведено 3 решения в виде сабмодулей:
- baseline на fasttext
- LSTM на allennlp 0.9.0
- fine-tune BERT от DeepPavlov

### Baseline

В качестве бейзлайна обучен fasttext с перебором гиперпараметров optuna.

Запуск обучения и получение предсказаний из текстового файла:
```bash
make train_fasttext
make predict_fasttext
```
или
```bash
python3 fasttext/train.py \
    --train_path 'fasttext/data/train.txt' \
    --valid_path 'fasttext/data/valid.txt' \
    --output_dir 'fasttext/data' \
    --n_trials 50

python3 fasttext/predict.py \
    --input_pat "fasttext/data/test.txt" \
    --output_path "fasttext/data/preds.txt" \ 
    --model_path "fasttext/model.fasstext"
```

### allennlp
LSTM на allennlp. В изначальной модели логировалась только метрика accuracy,
поэтому класс модели изменен так чтобы возвращалось F0.5.
Подключено логгирование в tensorboard.

Обучение в docker
```bash
make build_allennlp
make train_allennlp
```
или
```bash
nvidia-docker build allennlp -t classification
nvidia-docker run --name classification \
		--env DATA_DIR="/home/src/data" \
		--env BATCH_SIZE=32 \
		--env EPOCHS=10 \
		--env CUDA_DEVICES=2 \
	    -v $(pwd)/allennlp:/home/src classification
```

### BERT
fine-tuning bert от DeepPavlov. Трейнлуп на catalyst и он же логигирует в tensorboard
Для борьбы с дисбалансом классов вместо CrossEntropy взят FocalLoss.

Наибольший скор на валидации достигается при двухступенчатом обучении - сначала на даунсепленной выборке, затем дообучение на всей выборке. **F1 ~ 0.8**

Возможно обучение как на сервере с GPU, так и на slurm в docker:
```bash
make train_bert_classification
make predict_bert
make build_bert_classification
make train_bert_classification_slurm
```
или
```bash
python3 train.py
python3 predict.py \
    --input_path 'test.txt' \
    --output_path 'bert_preds.csv' \
    --model_path 'bert_for_classification.model' 
nvidia-docker build bert_classification -t bert_classification
nvidia-docker run --name bert_classification -v $(pwd)/bert_classification
```


### TBD
Что хотелось бы доделать:

 - [ ] Дообучение BERT-эмбеддингов по предложения (SBERT, DP--bert-base-cased-sentence) сиамской нейросетью и поиск ближайшего по косинусному расстоянию
 - [ ] Модуль make_dataset недоделан. Всю логику по работе с различными форматами датасетов можно вынести в этот модуль разгрузив train.
 - [ ] Метрики в отдельный модуль
 - [ ] Перебор гиперпараметров allennlp-optuna. Переход на версию 1.3
 - [ ] Видно что inference в predict.py для fasttext и bert имплементируют один интерфейс. Можно вынести логику в отдельный модуль Predictor который можно будет легко встраивать в сервисы, по необходимости
 - [ ] Аналогично, можно сделать модуль Trainer в который инжектить различные модели на обучение и запуская единным API, но пока кажется это нецелесообразно
 - [ ] Конфиги просто лежат ка словари в .py. Подгружать как json из /config
 - [ ] Реализовать Fbeta callback в catalyst
 - [ ] Написать docstring'и
 - [ ] Линтинг
 - [ ] click вместо argparse
 - [ ] ...
