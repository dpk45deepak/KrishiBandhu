# Examples

## Quick Start

1. Place data files in `data/raw/`.
2. Scan datasets:

```bash
python main.py scan
```

3. Profile datasets:

```bash
python main.py profile
```

4. Validate a dataset:

```bash
python main.py validate data/raw/my_dataset.csv --schema configs/crop_schema.yaml
```

5. Clean a dataset:

```bash
python main.py clean data/raw/my_dataset.csv --save-interim --report reports/cleaning
```

6. List profiling reports:

```bash
python main.py report
```

## Example Output

- Profiling reports are saved under `reports/profiling/`.
- Validation reports are saved under `reports/validation/`.
- Cleaned data can be saved to interim directories.

## Notes

- Supported dataset formats: CSV, XLS, XLSX, Parquet.
- Configuration is loaded from `configs/config.yaml`.
- The CLI uses `Rich` for styled console output.
