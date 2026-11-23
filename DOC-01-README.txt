Site geography and racial representation in type 2 diabetes trials

REQUIREMENTS
Python 3.11 or later with numpy, pandas, scipy, and pyarrow. Tested with Python 3.11.15,
NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1, and PyArrow 25.0.0.

HOW TO RUN
Put all files in one folder, open a terminal in that folder, and run:

  python CODE-01-reproduce.py    Every estimate from the trial data: Tables 1 to 3,
                                 Figures 1 and 2, Tables S1 to S4, and the post hoc
                                 analyses (sites in Asia, dedicated race tables, age
                                 under 18 years, US-only sites by state)
  python CODE-02-benchmarks.py   Rebuilds the US diabetes benchmarks (DATA-02) and their
                                 survey errors, checks them against DATA-02, and computes
                                 the alternative errors in Table S2

CODE-01 runs offline in about 10 seconds. CODE-02 downloads two public NHANES files from the
CDC website into a nhanes/ subfolder on first run. Each script writes its results as CSV
files in the same folder. The bootstrap is seeded from trial identifiers, so reruns give
identical output.

DATA SOURCES
ClinicalTrials.gov version 2 API: every record with posted results returned on September 25,
2026, for condition "diabetes mellitus type 2", interventional, randomized, phase 3 or 4
(including phase 2/3), start date on or after January 1, 2010. Baseline counts were retrieved
September 12, 2026. Eligibility ages (DATA-06) and US-only facility records (DATA-07) were
retrieved November 21, 2026.
NHANES 2017 to March 2020 prepandemic files (CDC).
American Community Survey 2019 to 2023 five-year tables (US Census Bureau).

NOTES
1. Read DATA-01 with keep_default_na=False, as the code does. An empty site_countries field
   means the trial posted no locations. Default pandas settings turn it into a missing value,
   which would wrongly classify these 12 trials as having non-US sites only.
2. The bootstrap resamples trials by position. The race_order and ethnicity_order columns fix
   that order, and the code sorts on them, so DATA-01 can be re-sorted without changing results.
3. Registry records can be edited after posting, so a new query may return different values.

LICENSE
Data and documentation: CC BY 4.0. Code: MIT.
