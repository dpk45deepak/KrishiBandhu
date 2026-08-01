import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.getcwd())

from app.data.profiling.profiler import DataProfiler
from app.data.profiling.report_generator import ReportGenerator

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / 'null_stats.csv'
    pd.DataFrame({'all_missing': [None, None, None], 'category': ['A', 'B', 'C']}).to_csv(p, index=False)
    profiler = DataProfiler(engine='pandas')
    result = profiler.profile(p)
    report_gen = ReportGenerator(output_dir=Path(tmp) / 'reports')
    paths = report_gen.generate_all(result)
    print('generated', paths)
    for key, path in paths.items():
        print(key, Path(path).exists(), Path(path).stat().st_size)
