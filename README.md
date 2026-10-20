# CSET485 – AI and Society · Assignment 1 · COMPAS stress-tests

Roll number 0450 (last digit 0 → COMPAS dataset). Random seed 450, 70/30 train–test split.

## Structure
```
notebook/CSET485_Assignment1_COMPAS.ipynb   all analysis; writes results/
src/report_text.py                          narrative text, numbers read from results.json
src/build_report.py                         builds report/CSET485_Assignment1_Report.pdf
src/verify_results.py                       independent re-computation (scikit-learn + numpy)
data/                                       compas-scores-two-years.csv (downloaded automatically if absent)
results/                                    results.json, tables/*.csv, figures/*.png
report/                                     final PDF
requirements.txt
```

## Setup
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce
1. `jupyter nbconvert --to notebook --execute --inplace notebook/CSET485_Assignment1_COMPAS.ipynb`
   (or open it and Run All). It downloads the dataset from
   https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv
   into `data/` if missing and records its SHA-256 in `results/results.json`.
2. `python src/verify_results.py`   (optional independent check)
3. `python src/build_report.py`     (the last notebook cell also does this if reportlab is installed)

## Notes
* The brief's `is_recidivist` column does not exist; it is defined as `two_year_recid` (see report §2).
* Fairness Group A = Caucasian, Group B = African-American (see report §8 for why).

## Publish to GitHub
```
git init && git add . && git commit -m "CSET485 Assignment 1"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```
Then open the repo URL in a private window to confirm it loads, and submit that link.
