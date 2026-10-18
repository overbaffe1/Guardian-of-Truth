# Удалённый прогон на GitHub-раннере

Песочница разработки видит только GitHub и PyPI, а данные соревнования лежат на
Sber OBS (presigned-ссылки, живут 24 часа). Поэтому тяжёлые и «внешние» вещи выполняются
на GitHub-hosted раннере, а в репозиторий возвращаются **только агрегированные метрики**.

## Как это устроено

```
локальная песочница            GitHub Actions (ubuntu-latest, 4 ядра, 86 ГБ)
        │                                    │
        │  repository_dispatch {url,label}   │
        ├───────────────────────────────────►│  curl URL            (внешняя сеть)
        │                                    │  pip install .
        │                                    │  .github/scripts/remote_eval.py
        │                                    │  → reports/<label>.json  (только метрики)
        │◄───────────────────────────────────┤  git push в ветку сессии
   git fetch / git show
```

Что попадает в отчёт (`reports/<label>.json`):

* `dataset` — число строк, баланс меток, распределение по доменам, квантили длин,
  наличие `<policy>` и `[AVAILABLE TOOLS]`, доля ответов с вызовами инструментов;
* `signals` — по каждому правилу: сколько раз сработало, precision, coverage, recall;
* `heuristic` — F1/P/R rule-based fallback целиком, confusion matrix, лучший порог;
* `file` — размер и sha256 скачанного файла.

**Сырые строки данных в репозиторий не попадают.** Это осознанно: репозиторий публичный,
а условие запрещает передачу данных третьим лицам. При необходимости файл можно
положить в артефакты (`store_raw=true`), но по умолчанию это выключено.

## Запуск

### 1. Через API (то, что делает агент)

```bash
gh api -X POST repos/overbaffe1/Guardian-of-Truth/dispatches \
  -f event_type=remote-eval \
  -F "client_payload[url]=<presigned link>" \
  -F "client_payload[label]=public"
```

`client_payload` не хранится ни в git, ни в файлах workflow; перед использованием ссылка
маскируется в логах (`::add-mask::`).

### 2. Через интерфейс GitHub

Actions → **remote-eval** → *Run workflow* → вставить ссылку и метку.
Этот способ работает потому, что workflow лежит в `main`: `repository_dispatch` и кнопка
запуска видны только для workflow из ветки по умолчанию.

### 3. Локально, если есть свой `gh` с правами на Actions

```bash
gh workflow run remote-eval.yml --ref arena/ff28bb09-guardian-of-truth \
  -f url="<presigned link>" -f label=public
```

## Чтение результата

Отчёт коммитится в ветку сессии, поэтому достаточно обновить git:

```bash
git fetch origin && git show origin/arena/ff28bb09-guardian-of-truth:reports/public.json
```

## Ограничения

* Раннеры **без GPU**, поэтому в отчёт попадает только rule-based часть и статистика.
  Полный LLM-судья проверяется на машине с A100 (см. `docs/CHECKLIST.md`).
* Лимит job — 60 минут; для файлов больше ~10 ГБ используйте `limit` и не включайте
  `store_raw`.
* Presigned-ссылки Sber OBS истекают через сутки — при ошибке 403 запросите новую.

## Самопроверка

При изменении `.github/workflows/remote-eval.yml` или `.github/scripts/remote_eval.py`
workflow запускается автоматически (push) и прогоняет маленький фикстур
`tests/fixtures/tau2_samples.jsonl`, складывая результат в `reports/fixture.json`.
Это проверяет установку пакета, метрики и коммит отчёта без внешних ссылок.
