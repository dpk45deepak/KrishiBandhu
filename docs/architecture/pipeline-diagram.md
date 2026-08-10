# Pipeline Diagram

```mermaid
flowchart LR
    A[Ingestion] --> B[Profiling]
    B --> C[Validation]
    C --> D[Cleaning]
    D --> E[Feature Engineering]
    E --> F[Model Training]
    F --> G[Reports]
```

This reflects the data-processing pipeline modules currently present under [core/app/data](../../app/data) and the ML/reporting subsystems under [core/app/ml](../../app/ml) and [core/app/services](../../app/services).
