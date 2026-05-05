# Data Codebook

This directory contains the panel data used in the empirical analysis. All files are CSV with comma-separated values, UTF-8 encoded. Variables follow standard econometric naming conventions.

## Files

| File | Rows | Description |
|------|------|-------------|
| `panel_main.csv` | 20,318 | Main estimation panel: firm-year observations of Chinese A-share listed firms (2007–2024) with non-missing dependent variable, supplier-base digital disparity, and the full control set including governance variables. Used for Tables 1, 2, 3, 5, 6 and Figure 5. |
| `panel_did.csv` | 33,534 | Broader firm-year panel with non-missing dependent variable and main controls (Size, Leverage, ROA). Used for the sanctions-based difference-in-differences (Table 4) and the dynamic event study (Figure 4). |
| `sanctions_list.csv` | 24 | Treated firms in the sanctions-based DID, with first U.S. designation year. |
| `supplier_network_examples.csv` | 586 | First- and second-order supplier network for four representative focal firms in 2024, with supplier-level patent-based digital intensity. Used for Figure 3. |

## Variables in `panel_main.csv` and `panel_did.csv`

### Identifiers and panel structure

| Column | Type | Description |
|--------|------|-------------|
| `stkcd` | string (6 digits) | Stock code identifier of the focal firm. |
| `year` | int | Calendar year (2007–2024). |
| `industry_code` | string | CSRC industry classification code. |

### Outcome

| Column | Type | Description |
|--------|------|-------------|
| `resilience_composite` | float | Firm-year supply chain resilience composite computed from financial-statement data. Combines two resistance sub-indices (sign-reversed accounts-receivable-to-revenue ratio; share of top-five customers retained from the prior year) and two recovery sub-indices (negative production-to-demand variance ratio, with production reconstructed from inventory dynamics and cost of goods sold; residual from a regression of EBIT per employee on firm size, leverage, growth, age, board size, firm fixed effects, and year fixed effects) through annual entropy weighting after within-year min-max normalization. Higher values indicate stronger resilience. |

### Main explanatory variable

| Column | Type | Description |
|--------|------|-------------|
| `dd_patent_range` | float | Supplier-base digital disparity (primary measure). Range of patent-based digital intensity across the focal firm's first- and second-order suppliers in a given year. Patent-based digital intensity is the share of digital patents (IPC categories G06F, H04L, G06Q, G06N, G06K) in a supplier's total invention patents. Industry-year mean values are imputed for unlisted suppliers. |

### Alternative disparity measures (for robustness)

| Column | Type | Description |
|--------|------|-------------|
| `dd_recruitment_range` | float | Range of recruitment-based digital intensity across suppliers (digital job postings as a share of total postings). |
| `dd_composite_range` | float | Range of the composite digital intensity (equally-weighted average of patent and recruitment ratios). |
| `dd_patent_cv` | float | Coefficient of variation of supplier patent-based digital intensity. |
| `dd_patent_std` | float | Standard deviation of supplier patent-based digital intensity. |
| `n_suppliers_observed` | float | Number of suppliers in the focal firm's observed network (used for diagnostics). |

### Controls

| Column | Type | Description |
|--------|------|-------------|
| `size` | float | Firm size: natural logarithm of total assets. |
| `leverage` | float | Total liabilities divided by total assets. |
| `roa` | float | Return on assets: net income divided by total assets. |
| `top1_ownership` | float | Shareholding ratio of the largest shareholder. |
| `board_size` | float | Number of directors on the board. |
| `board_independence` | float | Number of independent directors divided by total directors. |
| `industry_hhi` | float | Industry concentration measure (Herfindahl–Hirschman Index of industry sales). |
| `firm_age` | float | Years since establishment. |
| `list_age` | float | Years since IPO. |
| `is_high_tech` | int (0/1) | Indicator for high-technology industry. |

### Moderators

| Column | Type | Description |
|--------|------|-------------|
| `geo_dispersion_count` | float | Number of distinct provinces in which the focal firm's suppliers are located. |
| `sa_index` | float | SA index of financial constraints (Hadlock and Pierce, 2010). Higher values indicate tighter constraints. |

### Treatment (DID)

| Column | Type | Description |
|--------|------|-------------|
| `first_sanction_year` | float (NA for control firms) | First calendar year the focal firm was named on a major U.S. trade or investment restriction list (Entity List or NS-CMIC). |

## Variables in `sanctions_list.csv`

| Column | Type | Description |
|--------|------|-------------|
| `stkcd` | string | Stock code of the treated firm. |
| `first_sanction_year` | int | First U.S. sanction year. |

## Variables in `supplier_network_examples.csv`

| Column | Type | Description |
|--------|------|-------------|
| `focal_stkcd` | string | Stock code of the focal listed firm (one of four representative cases). |
| `year` | int | Year (all 2024 in this file). |
| `supplier_stkcd` | string | Stock code or registry identifier of a supplier node. |
| `supplier_is_listed` | int (0/1) | Whether the supplier is itself an A-share listed firm. |
| `node_type` | string | "supplier" for all rows in this file. |
| `patent_digital_ratio` | float | Patent-based digital intensity of the supplier (NA for unlisted suppliers without observable patents). |

## Notes on data construction

- The supplier network is reconstructed from Tianyancha relationship records (public bidding announcements and contract awards) supplemented by annual-report supplier disclosures.
- Continuous variables in regressions are winsorized at the 1st and 99th percentiles within sample, then standardized to within-sample Z-scores. Reported coefficients are interpretable as standard-deviation changes.
- Industry-year mean digitalization values are assigned to unlisted suppliers without observable patent data. This imputation compresses cross-supplier heterogeneity and renders the range-based disparity measure conservative.
- The 33,534 raw panel and the 20,318 main estimation sample differ in the set of required non-missing variables. The 20,318 sample additionally requires Top-1 ownership, board size, board independence, and industry HHI to be observed.
