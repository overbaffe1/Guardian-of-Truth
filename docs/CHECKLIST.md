# Чек-лист на день соревнования

## 0. Окружение

* GPU: 1× A100 (40 или 80 ГБ), CUDA-драйвер совместим с образом `vllm/vllm-openai:v0.6.6`.
* Образ собран заранее, веса лежат в `/app/model` (см. `model/README.md`).
* Проверка: `python -c "import vllm, guardian; print(vllm.__version__)"`.
* Полный pre-flight (GPU, пакеты, веса, прогон 8 строк и прогноз времени на 543):
  ```bash
  python scripts/doctor.py --input public.parquet --limit 8
  ```
  В отчёте смотреть `stats.backend` (должен быть `vllm`, не `heuristic`),
  `projected_sec_private` (< 1500) и `warnings`.

## 1. Публичный сплит (без спешки)

```bash
# 1.1 прогон с деталями
python scripts/predict.py \
    --input public.parquet --output /tmp/public.parquet --with-details

# 1.2 метрики + диагностика сигналов
python scripts/evaluate.py --data public.parquet --threshold 0.5
python scripts/calibrate.py --preds /tmp/public.parquet --truth public.parquet \
    --out configs/calibrated.json
```

Что смотреть:

* `metrics.f1` — целевая метрика;
* `judge_only` — насколько хорош сам судья без правил;
* `signals.*.precision` — не даёт ли какой-то hard-сигнал ложных срабатываний
  (тогда `identifier_severity: "soft"` или `rule_mode: "hint"`);
* `best_threshold` — порог, который сохранить в `configs/calibrated.json`;
* `elapsed_sec` — запас по времени (нужно < 1500 c на ~550 примеров);
* `verified` / `verify_overrides` — сколько отрицательных прошло второе мнение и сколько
  из них перевернулось в `1`; если `verify_overrides` близко к нулю, второй проход можно
  отключить и сэкономить время.

## 2. Приватный сплит

```bash
python scripts/predict.py \
    --input private.parquet --output /tmp/private.parquet \
    --config configs/calibrated.json --with-details

python scripts/check_submission.py --input private.parquet --preds /tmp/private.parquet
```

Второй прогон «на всякий случай» с усреднением можно собрать как ансамбль двух файлов:
`python scripts/blend.py --preds a.parquet b.parquet --out c.parquet --how vote`.

Проверить:

* `positive_rate` не 0 и не 1;
* `backend` = `vllm` (а не `heuristic`!);
* `elapsed_sec` < 1500;
* файл содержит ровно `id`, `label`.

## 3. Если что-то пошло не так

| Симптом | Действие |
|---|---|
| vLLM не поднялся | `--backend transformers --model ...` (медленнее) или `--backend heuristic` (только правила) |
| Не хватает времени | `--max-prompt-tokens 8000`, `--self-consistency 1`, модель 7B |
| Слишком много ложных `1` | поднять `--threshold` до 0.6–0.7, `identifier_severity: "soft"` |
| Слишком мало `1` | снизить порог до 0.35–0.4, включить `soft_signal_override` |
| Долгая загрузка модели | кэшировать образ, веса уже внутри |
| OOM | `--set model.gpu_memory_utilization=0.85 --set model.max_model_len=16384` |
| Второй проход не успевает | `--set judge.verify_negatives=false` (потеря ~нескольких recall-пунктов) |
| Нужно быстро сравнить варианты | `--set judge.verify_threshold=0.6`, `--set fusion.rule_mode=hint` |

## 4. Сабмит

* Убедиться, что в образе нет датасетов и ответов (`.dockerignore`).
* Размер образа < 40 ГБ: `docker images guardian-of-truth`.
* Выбрать 3 итоговых решения в интерфейсе конкурса (если загружено больше).
