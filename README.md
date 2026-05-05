# Uneven Supplier Digitalization, Dependence Reconfiguration, and Supply Chain Resilience — Replication Package

Panel data and analysis code for the empirical results in the paper. The
sample is a panel of Chinese A-share listed firms from 2007 to 2024.

## Repository structure

```
replication_package/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── panel_main.csv
│   ├── panel_did.csv
│   ├── sanctions_list.csv
│   ├── supplier_network_examples.csv
│   └── codebook.md
└── code/
    ├── utils.py
    ├── 01_descriptive.py
    ├── 02_baseline.py
    ├── 03_iv.py
    ├── 04_did.py
    ├── 05_moderation.py
    ├── 06_mechanism.py
    ├── 07_robustness.py
    └── 08_figures.py
```

## Quick start

```bash
pip install -r requirements.txt
cd code
python 01_descriptive.py   # Table 1
python 02_baseline.py      # Table 2
python 03_iv.py            # Table 3
python 04_did.py           # Table 4 + Figure 4
python 05_moderation.py    # Table 5 + Figure 5
python 06_mechanism.py     # Table 6
python 07_robustness.py    # Selected appendix robustness rows
python 08_figures.py       # Figures 2 and 3
```

Each script reads from `data/` and writes its output to `output/`
(gitignored).

Variable definitions are in `data/codebook.md`. Each script's docstring
describes its specification. Continuous variables in `panel_main.csv` are
pre-winsorised at the 1st and 99th percentiles. Standard errors are
clustered at the firm level throughout.

## License

Released under the MIT License. See `LICENSE`.
