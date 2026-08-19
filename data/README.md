# Data

The original project used real Siemens SENTRON PAC4200 measurements and generated
synthetic training/test data.

The public portfolio repository intentionally does **not** include the full raw
or synthetic datasets. This keeps the repository lightweight and avoids
redistributing university/project measurement data.

The notebooks and source code document the processing pipeline and the committed
`outputs/reports/` files preserve the key classification results.

Expected local folders when reproducing the full pipeline:

```text
data/
├── raw/
├── clean/
└── synthetic/
    ├── training/
    └── test/
```
