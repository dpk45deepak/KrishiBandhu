# Folder Structure

## Repository layout

```text
core/
├── app/
│   ├── cli/
│   ├── config/
│   ├── constants/
│   ├── core/
│   ├── data/
│   ├── logger/
│   ├── ml/
│   ├── services/
│   └── utils/
├── configs/
├── data/
├── docs/
├── frontend/
├── logs/
├── models/
├── reports/
├── scripts/
├── tests/
└── main.py
```

## Package responsibilities

- [core/app/cli](../../app/cli): command-line entrypoints and command registration.
- [core/app/config](../../app/config): Pydantic settings and YAML-backed config loading.
- [core/app/core](../../app/core): runtime integration layer, container, event bus, lifecycle hooks, and plugin abstractions.
- [core/app/data](../../app/data): data processing capabilities such as profiling, validation, feature engineering, and pipeline orchestration.
- [core/app/ml](../../app/ml): model training, evaluation, explainability, and tuning.
- [core/app/services](../../app/services): higher-level service objects for API-driven workflows.
- [core/app/utils](../../app/utils): common utilities such as path handling and decorators.
