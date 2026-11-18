# Запуск и проверка Guardian of Truth

Для первого запуска используйте [CPU-демо](../README.md#запуск). Все команды ниже рассчитаны на запуск из корня репозитория; `public.parquet` и `test.parquet` обозначают ваши файлы данных.

## Инференс с моделью

Для локального запуска на совместимой с vLLM/CUDA машине установите зависимости и скачайте веса (около 9 ГБ). По умолчанию используется `Qwen/Qwen2.5-14B-Instruct-AWQ`.

```bash
python -m pip install '.[llm]'
python scripts/download_model.py --out model
python scripts/predict.py --backend vllm --model-path model --input test.parquet --output predictions.parquet
```

Проверьте, что в итоговой сводке `backend` — `vllm`, а не `heuristic`: при ошибке загрузки детектор может продолжить без модели. Параметры памяти, размера батча и бюджета времени собраны в [`configs/default.json`](../configs/default.json). `--set KEY=VALUE` переопределяет отдельный ключ, например `--set judge.verify_negatives=false`.

### Docker

```bash
docker build -t guardian-of-truth .
docker run --rm --gpus all -v "$PWD/data:/data" guardian-of-truth \
  python scripts/predict.py --input /data/test.parquet --output /data/predictions.parquet --backend vllm
```

Поместите `test.parquet` в локальный каталог `data/` перед запуском. Сборка по умолчанию скачивает веса. Если они уже лежат в `model/`, добавьте `--build-arg SKIP_MODEL_DOWNLOAD=1`; `--build-arg MODEL_ID=...` задаёт альтернативную модель. См. [каталог модели](../model/README.md) и [чек-лист соревнования](CHECKLIST.md).

## Диагностика и калибровка

```bash
python scripts/inspect_example.py --input test.parquet --index 0 --show-prompt
python scripts/predict.py --input public.parquet --output public-diagnostics.parquet --with-details
python scripts/calibrate.py --preds public-diagnostics.parquet --truth public.parquet --out configs/calibrated.json
```

`--limit N` ограничивает число записей, `--debug-jsonl PATH` сохраняет построчную диагностику. Файл с `--with-details` содержит дополнительные колонки (`score`, `judge_label`, `judge_confidence`, `signals`, `reason`) и предназначен для анализа, а не для сдачи. Для финального прогона передайте `--config configs/calibrated.json` и уберите `--with-details`.

Чтобы измерить метрики на размеченном наборе без подбора порога, запустите `python scripts/evaluate.py --data public.parquet`. Добавьте `--backend heuristic` для прогона только по правилам. [Сохранённые CPU-замеры](../reports/README.md) не заменяют отдельную проверку модели.

<details>
<summary>Собрать синтетический набор из tau2-bench</summary>

```bash
git clone --depth 1 https://github.com/sierra-research/tau2-bench /tmp/tau2
python scripts/make_dev_from_tau2.py --tau2-root /tmp/tau2 --out data/dev_synthetic.jsonl --per-domain 40
python scripts/evaluate.py --data data/dev_synthetic.jsonl --backend heuristic
```

Успешные траектории дают примеры с меткой `0`. Контролируемые подмены сумм, идентификаторов, фактов и инструментов добавляют примеры с меткой `1`. Это dev-набор, а не официальная разметка.

</details>

## Тесты и упаковка

```bash
python -m pip install -e '.[dev]'
python -m pytest
python scripts/check_submission.py --input test.parquet --preds predictions.parquet
python scripts/make_submission.py --check
python scripts/make_submission.py --out submission.zip
```

`check_submission.py` проверяет количество строк, совпадение идентификаторов входного и выходного Parquet и допустимость меток. `make_submission.py --check` ожидает локальные веса в `model/` и сообщит, если их нет. Для офлайн-обсчёта веса должны быть в архиве; детали формата — в [требованиях к сабмиту](SUBMISSION.md).
