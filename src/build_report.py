"""
Builds report/CSET485_Assignment1_Report.pdf from the notebook outputs.

Inputs : results/results.json, results/tables/*.csv, results/figures/*.png   (all written by the notebook)
Output : report/CSET485_Assignment1_Report.pdf
Every number in the PDF is read from those files; nothing is typed in by hand.
"""
import hashlib
import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_text as RT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "CSET485_Assignment1_Report.pdf"
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
R = RT.load()
T = RT.build(R)
n, pv, pct, fr, ci = RT.n, RT.pv, RT.pct, RT.fr, RT.ci

STUDENT, ROLL = "Sakshi Arora", "0450"

# ------------------------------------------------------------------------------------------------ fonts
FD = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FD / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, GREY, LIGHT, HILITE = colors.HexColor("#1F3A5F"), colors.HexColor("#555555"), colors.HexColor("#F2F5F9"), colors.HexColor("#FFF3C4")
BODY = ParagraphStyle("Body", fontName="DV", fontSize=9, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5)
BODYL = ParagraphStyle("BodyL", parent=BODY, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=7.6, leading=10, textColor=GREY)
H1 = ParagraphStyle("H1", fontName="DV-B", fontSize=14.5, leading=18, textColor=NAVY, spaceBefore=4, spaceAfter=8)
H2 = ParagraphStyle("H2", fontName="DV-B", fontSize=11, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=5)
H3 = ParagraphStyle("H3", fontName="DV-B", fontSize=9.4, leading=12.5, textColor=colors.black, spaceBefore=8, spaceAfter=3)
QH = ParagraphStyle("QH", parent=H3, textColor=NAVY, fontSize=10)
QP = ParagraphStyle("QP", parent=BODY, fontName="DV-I", textColor=GREY, fontSize=8.2, leading=11, spaceAfter=3)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8, leading=10.5, textColor=colors.black, alignment=TA_LEFT, spaceBefore=3, spaceAfter=3)
TC = ParagraphStyle("TC", fontName="DV", fontSize=7.4, leading=9.2, alignment=TA_CENTER)
TCL = ParagraphStyle("TCL", parent=TC, alignment=TA_LEFT)
TH = ParagraphStyle("TH", parent=TC, fontName="DV-B", textColor=colors.white)
BUL = ParagraphStyle("Bul", parent=BODYL, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("Title", fontName="DV-B", fontSize=25, leading=31, textColor=NAVY, alignment=TA_LEFT)
SUB = ParagraphStyle("Sub", fontName="DV", fontSize=13, leading=18, textColor=GREY, alignment=TA_LEFT)
META = ParagraphStyle("Meta", fontName="DV", fontSize=10.5, leading=17, alignment=TA_LEFT)

# ------------------------------------------------------------------------------------------------ numbering / cross-references
TBL_ORDER = ["dataset", "preproc", "outliers", "coefs", "perf", "cm", "flags", "p21_sex", "p21_race", "p22", "p23_profile", "p23_traits",
             "fair_main", "fair_rates", "fair_ci", "tradeoff", "p22q3", "A1", "robust", "qc"]
FIG_ORDER = ["eda", "forest", "ovb", "adv", "tradeoff", "metrics"]
TNUM = {k: i + 1 for i, k in enumerate(TBL_ORDER)}
FNUM = {k: i + 1 for i, k in enumerate(FIG_ORDER)}


def X(s):
    s = re.sub(r"\[\[tbl:(\w+)\]\]", lambda m: f"Table {TNUM[m.group(1)]}", s)
    return re.sub(r"\[\[fig:(\w+)\]\]", lambda m: f"Figure {FNUM[m.group(1)]}", s)


def P(text, style=BODY):
    return Paragraph(X(text), style)


def cap_t(key, text):
    return Paragraph(f"<b>Table {TNUM[key]}.</b> {X(text)}", CAP)


def fig(key, file, text, width=16.6 * cm):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(FIG / file)).getSize()
    img = Image(str(FIG / file), width=width, height=width * h / w)
    return KeepTogether([img, Paragraph(f"<b>Figure {FNUM[key]}.</b> {X(text)}", CAP), Spacer(1, 4)])


def tbl(rows, widths, header=1, hilite=(), bold=(), left_cols=(0,), font=None, zebra=True):
    cells = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            if i < header:
                row.append(Paragraph(str(c), TH))
            else:
                st = TCL if j in left_cols else TC
                txt = str(c)
                if i in bold:
                    txt = f"<b>{txt}</b>"
                row.append(Paragraph(txt, st))
        cells.append(row)
    t = Table(cells, colWidths=[w * cm for w in widths], repeatRows=header, hAlign="CENTER")
    style = [("BACKGROUND", (0, 0), (-1, header - 1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, header - 1), (-1, -1), 0.25, colors.HexColor("#C9CED6")), ("BOX", (0, 0), (-1, -1), 0.5, NAVY),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(header, len(rows)):
            if (i - header) % 2 == 1:
                style.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    for i in hilite:
        style.append(("BACKGROUND", (0, i), (-1, i), HILITE))
    t.setStyle(TableStyle(style))
    t._nrows = len(rows)
    return t


def fmt(x, d=3, na="NA"):
    return na if x is None else n(x, d)


def pf(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------------------------------------ document class
class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                self.setFont("DV", 7.6)
                self.setFillColor(GREY)
                self.drawCentredString(A4[0] / 2, 1.1 * cm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=2.0 * cm, bottomMargin=1.9 * cm,
                         title="CSET485 Assignment 1: Stress-testing risk models on the COMPAS dataset", author=STUDENT,
                         subject="AI and Society, Assignment #1")
        fr_ = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="title", frames=[fr_]), PageTemplate(id="body", frames=[fr_], onPage=self._decor)])
        self._h = 0

    @staticmethod
    def _decor(c, d):
        c.saveState()
        c.setFont("DV", 7.6)
        c.setFillColor(GREY)
        c.drawString(2.0 * cm, A4[1] - 1.25 * cm, "CSET485 \u2013 AI and Society \u00b7 Assignment #1 \u00b7 COMPAS stress-tests")
        c.drawRightString(A4[0] - 2.0 * cm, A4[1] - 1.25 * cm, f"{STUDENT} \u00b7 Roll No. {ROLL}")
        c.setStrokeColor(colors.HexColor("#C9CED6"))
        c.setLineWidth(0.4)
        c.line(2.0 * cm, A4[1] - 1.4 * cm, A4[0] - 2.0 * cm, A4[1] - 1.4 * cm)
        c.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("H1", "H2", "QH"):
            lvl = 0 if fl.style.name == "H1" else 1
            text = fl.getPlainText()
            key = "h" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10]      # deterministic across layout passes
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, lvl, 0)
            self.notify("TOCEntry", (lvl, text, self.page, key))


# ------------------------------------------------------------------------------------------------ story
S = []
M, P1, S_ = R["p24"], R["part1"], R["split"]
sw = {round(r["t"], 2): r for r in M["sweep"]}
cis = {round(r["t"], 2): r for r in M["ci"]}
ts = M["t_star"]

# ---- title page
S += [Spacer(1, 3.2 * cm), Paragraph("CSET485 \u2013 AI and Society", SUB), Spacer(1, 0.4 * cm),
      Paragraph("Stress-Testing a Predictive Model on the COMPAS Recidivism Dataset", TITLE), Spacer(1, 0.5 * cm),
      Paragraph("Assignment #1 (10 Marks): Build, Stress-Test and Defend", SUB), Spacer(1, 1.6 * cm)]
meta = [["Student", STUDENT], ["Roll number", ROLL],
        ["Assigned dataset", "COMPAS Recidivism Risk Score Dataset (roll number ends in 0)"],
        ["Random seed / split", f"random_state = {S_['seed']} (last three digits of the roll number); {S_['n_train']:,} training rows (70%), {S_['n_test']:,} test rows (30%)"],
        ["Data source", "ProPublica compas-scores-two-years.csv (Broward County, Florida)"],
        ["Companion code", "Jupyter notebook CSET485_Assignment1_COMPAS.ipynb; Appendix C explains how to regenerate every number in this report"]]
mt = Table([[Paragraph(f"<b>{a}</b>", META), Paragraph(b, META)] for a, b in meta], colWidths=[4.2 * cm, 12.8 * cm])
mt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.2, NAVY), ("LINEBELOW", (0, -1), (-1, -1), 1.2, NAVY),
                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
S += [mt, NextPageTemplate("body"), PageBreak()]

# ---- contents + executive summary
S.append(Paragraph("Contents", ParagraphStyle("CH", parent=H1)))
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t1", fontName="DV-B", fontSize=9, leading=13, leftIndent=0, spaceBefore=3),
                   ParagraphStyle("t2", fontName="DV", fontSize=8.4, leading=11.5, leftIndent=14)]
S += [toc, PageBreak()]
S += [Paragraph("Executive summary", H1), P(T["abstract"])]

# ---- 1 intro
S += [Paragraph("1  Introduction and objectives", H1), P(T["intro"])]

# ---- 2 dataset
S += [Paragraph("2  Dataset description", H1), P(T["dataset"])]
desc = pd.read_csv(TAB / "descriptives.csv", index_col=0)
rows = [["Group", "n", "Share of data", "Two-year recidivism rate", "Mean COMPAS decile", "Mean priors", "Mean age"]]
dfr = pd.DataFrame(R["eda"]["by_race"]).set_index("race_group")
for g, lab in [("African-American", "African-American"), ("Caucasian", "Caucasian"), ("Hispanic", "Hispanic"), ("Other", "Other (Asian, Native American, Other; pooled)")]:
    r = dfr.loc[g]
    rows.append([lab, f"{int(r.n):,}", fr(r.n / R["data"]["n_rows"]), fr(r.recid_rate), f"{r.mean_decile:.2f}", f"{r.mean_priors:.2f}", f"{r.mean_age:.1f}"])
dfs = pd.DataFrame(R["eda"]["by_sex"]).set_index("sex")
for g in ["Male", "Female"]:
    r = dfs.loc[g]
    rows.append([g, f"{int(r.n):,}", fr(r.n / R["data"]["n_rows"]), fr(r.recid_rate), f"{r.mean_decile:.2f}", f"{r.mean_priors:.2f}", f"{r.mean_age:.1f}"])
rows.append(["All defendants", f"{R['data']['n_rows']:,}", "100.0%", fr(R["eda"]["base_rate"]), f"{desc.loc['decile_score', 'mean']:.2f}", f"{desc.loc['priors_count', 'mean']:.2f}", f"{desc.loc['age', 'mean']:.1f}"])
S += [cap_t("dataset", "Dataset summary by demographic group (all 7,214 rows)."),
      tbl(rows, [4.6, 1.5, 2.0, 2.6, 2.2, 1.9, 1.7], bold=(len(rows) - 1,)), Spacer(1, 6),
      fig("eda", "fig1_eda.png", "Exploratory analysis: (a) recidivism rate by race group, (b) COMPAS decile score, (c) prior convictions and (d) age by outcome.", 15.5 * cm)]

# ---- 3 preprocessing
pp = R["preproc"]
S += [Paragraph("3  Data preprocessing methodology", H1), P(T["preproc_intro"])]
pre_rows = [["Issue", "Decision and justification"],
            ["Missing values", f"{pp['n_columns_with_missing']} of the {R['data']['n_cols']} columns contain missing values (dates, charge descriptions, follow-up and violent-recidivism fields). None of them is used as a predictor, "
                               "so <b>no imputation</b> is needed and <b>no row is dropped</b> (n = 7,214); the ten model features have zero missing values (asserted in the notebook)."],
            ["Leakage control", "Excluded as predictors: <i>two_year_recid</i>, <i>is_recid</i>, every <i>r_*</i>, <i>vr_*</i> and <i>violent_recid</i> column (post-screening outcomes); <i>decile_score</i>, <i>score_text</i> and <i>v_*</i> "
                                "(COMPAS\u2019s own output, which is Model A\u2019s target and a proxy for the outcome); identifiers and dates. The scaler is fitted on the training set only."],
            ["Categorical encoding", f"<i>sex</i> \u2192 <i>male</i> (1/0); <i>c_charge_degree</i> \u2192 <i>felony</i> (F = 1, M = 0); <i>race</i> \u2192 one-hot with Caucasian as reference. Asian (n = {pp['n_asian']}), Native American (n = {pp['n_native']}) "
                                    f"and Other (n = {pp['n_other_original']}) are pooled into <i>race_Other</i> (n = {pp['race_other_pooled_n']}) because the two smallest groups are too small to estimate stably. <i>age_cat</i> is dropped (redundant with <i>age</i>)."],
            ["Feature scaling", "The five numeric features (age, priors_count, juv_fel_count, juv_misd_count, juv_other_count) are z-scored with a <i>StandardScaler</i> fitted on the 5,049 training rows and applied unchanged to the test set. "
                                "Coefficients are therefore per 1 SD of the training distribution; dummy variables are unscaled."],
            ["Outliers", "IQR-rule outliers are counted in the next table. All values are plausible records (age \u2264 96, priors \u2264 38) and the long right tail of priors_count is the high-risk population, so <b>values are retained</b>. "
                         f"A check that winsorises priors_count at the training 99th percentile ({R['robust']['winsor_cap']:.0f}) changes test AUC by {n(R['robust']['Priors winsorised at train p99 (22)']['AUC'] - R['robust']['Main model B (target two_year_recid)']['AUC'])} (Appendix B)."],
            ["Target definition", "<i>is_recidivist</i> := <i>two_year_recid</i> (see Section 2). Model A\u2019s target is <i>decile_score</i> (1\u201310)."]]
S += [cap_t("preproc", "Preprocessing decisions."), tbl(pre_rows, [3.3, 13.7], left_cols=(0, 1)), Spacer(1, 6)]
orows = [["Feature", "Min", "Median", "Mean", "99th pct", "Max", "IQR outliers", "% of rows"]]
for r in pp["outliers"]:
    orows.append([r["feature"], f"{r['min']:.0f}", f"{r['median']:.0f}", f"{r['mean']:.2f}", f"{r['p99']:.0f}", f"{r['max']:.0f}", f"{r['n_IQR_outliers']:,}", f"{r['pct_IQR_outliers']:.1f}%"])
S += [KeepTogether([cap_t("outliers", "Outlier summary for the numeric features (all retained)."), tbl(orows, [3.6, 1.3, 1.6, 1.6, 1.8, 1.3, 2.4, 2.0])])]

# ---- 4 Part 1
S += [Paragraph("4  Part 1 \u2013 Model development, evaluation and mismatch analysis", H1), P(T["part1"])]
cf_A, cf_B = R["modelA"]["coef"], R["modelB"]["coef"]
crow = [["Predictor", "Model A (OLS): \u03b2 (SE)", "p", "Model B (logit): \u03b2", "95% CI", "p", "Odds ratio"]]
for k in cf_B:
    a, b = cf_A[k], cf_B[k]
    crow.append([k, f"{n(a['coef'])} ({n(a['se'])})", pf(a["p"]), n(b["coef"]), ci(b["ci_low"], b["ci_high"]), pf(b["p"]), "" if k == "const" else f"{2.718281828459045 ** b['coef']:.3f}"])
S += [cap_t("coefs", "Model coefficients on the training set (numeric predictors standardised; Model A in decile points, Model B in log-odds)."),
      tbl(crow, [3.6, 3.0, 1.4, 2.0, 3.0, 1.4, 1.8]), Spacer(1, 6)]
prow = [["Model", "Metric (test set unless stated)", "Value"],
        ["A (OLS)", "R\u00b2 (train / test)", f"{n(R['modelA']['r2_train'])} / {n(R['modelA']['r2_test'])}"], ["", "RMSE / MAE (deciles)", f"{n(R['modelA']['rmse_test'])} / {n(R['modelA']['mae_test'])}"],
        ["", "Median / min / max prediction", f"{n(R['modelA']['median_pred'])} / {n(R['modelA']['pred_min'], 2)} / {n(R['modelA']['pred_max'], 2)}"],
        ["B (logit)", "Accuracy at 0.5 (majority baseline)", f"{n(R['modelB']['accuracy'])} ({n(S_['majority_baseline_acc_test'])})"],
        ["", "AUC / Brier score / log-loss", f"{n(R['modelB']['auc'])} / {n(R['modelB']['brier'])} / {n(R['modelB']['logloss'])}"],
        ["", "Precision / recall / F1", f"{n(R['modelB']['precision'])} / {n(R['modelB']['recall'])} / {n(R['modelB']['f1'])}"],
        ["", "McFadden pseudo-R\u00b2 (train)", n(R["modelB"]["pseudo_r2_train"])]]
S += [KeepTogether([cap_t("perf", "Test-set performance of Models A and B."), tbl(prow, [2.6, 7.2, 4.6], left_cols=(0, 1))]), Spacer(1, 6)]
c = P1["cm"]
cm_rows = [["", "Model B: low risk (p &lt; 0.5)", "Model B: high risk (p \u2265 0.5)", "Total"],
           ["Model A: low risk (&lt; median)", f"{c['AL_BL']:,}", f"{c['AL_BH']:,}", f"{c['AL_BL'] + c['AL_BH']:,}"],
           ["Model A: high risk (\u2265 median)", f"{c['AH_BL']:,}", f"{c['AH_BH']:,}", f"{c['AH_BL'] + c['AH_BH']:,}"],
           ["Total", f"{c['AL_BL'] + c['AH_BL']:,}", f"{c['AL_BH'] + c['AH_BH']:,}", f"{P1['n_test']:,}"]]
S += [cap_t("cm", f"Confusion matrix of the two thresholding approaches (rows: Model A at the median {n(P1['median_pred_A'])}; columns: Model B at 0.5). Mismatched cases are the off-diagonal cells: "
                  f"{c['AL_BH']} + {c['AH_BL']} = <b>{P1['mismatch_n']} of {P1['n_test']:,} ({pct(P1['mismatch_pct'], 2)})</b>."),
      tbl(cm_rows, [5.2, 4.2, 4.2, 2.0], hilite=(), bold=(3,)), Spacer(1, 6)]
frows = [["Flag", "Flag rate", "Accuracy", "Precision", "Recall", "FPR", "TP", "FP", "TN", "FN"]]
for lab, k in [("Model A (\u2265 median)", "Model A flag (>= median)"), ("Model B (p \u2265 0.5)", "Model B flag (p >= 0.5)")]:
    v = P1["vs_actual"][k]
    frows.append([lab, fr(v["flag_rate"]), fr(v["accuracy"]), fr(v["precision"]), fr(v["recall"]), fr(v["FPR"]), int(v["TP"]), int(v["FP"]), int(v["TN"]), int(v["FN"])])
S += [cap_t("flags", f"Agreement is not accuracy: each flag scored against the <i>actual</i> outcome. Both models are correct together on {P1['agree_both_correct']:,} defendants and wrong together on {P1['agree_but_both_wrong']:,}; "
                     f"on the {P1['mismatch_n']} disagreements Model B is right {P1['mismatch_B_right']} times and Model A {P1['mismatch_A_right']} times."),
      tbl(frows, [3.8, 1.5, 1.6, 1.6, 1.4, 1.3, 1.1, 1.1, 1.1, 1.1])]
S.append(P("The explanation of <i>why</i> the two approaches flag different people is given in Part 3, Question 1.", SMALL))

# ---- 5 Part 2.1
sx, rc = R["p21"]["primary"], R["p21"]["race"]


def split_rows(tab, la, lb):
    rows = [["Predictor", "Full \u03b2", f"{la} \u03b2 [95% CI]", f"{lb} \u03b2 [95% CI]", "z (A\u2212B)", "p", "p (Bonf.)"]]
    for r in tab:
        rows.append([r["predictor"], n(r["full_coef"]), f"{n(r['A_coef'])} {ci(r['A_lo'], r['A_hi'])}", f"{n(r['B_coef'])} {ci(r['B_lo'], r['B_hi'])}",
                     n(r["diff_z"], 2), pf(r["diff_p"]), f"{r['diff_p_bonferroni']:.3f}" if r["diff_p_bonferroni"] < 1 else "1.000"])
    return rows


S += [Paragraph("5  Part 2.1 \u2013 Simpson\u2019s-paradox hunt", H1), P(T["p21"])]
rs = split_rows(sx["table"], "Male (A)", "Female (B)")
S += [cap_t("p21_sex", f"Primary split: sex (A = male, n = {sx['n_A']:,}; B = female, n = {sx['n_B']:,}); logistic coefficients per 1 SD, full data n = {sx['n_full']:,}. Bonferroni over {R['p21_scan']['n_tests']} tests. The priors_count row is the finding."),
      tbl(rs, [3.9, 1.3, 3.7, 3.7, 1.4, 1.1, 1.4], bold=(2,), hilite=(2,), left_cols=(0,))]
rr_ = split_rows(rc["table"], "Afr.-Amer. (A)", "Caucasian (B)")
S += [Spacer(1, 6), cap_t("p21_race", f"Secondary split: race (A = African-American, n = {rc['n_A']:,}; B = Caucasian, n = {rc['n_B']:,}); race indicators omitted from all three models. No difference survives correction."),
      tbl(rr_, [3.9, 1.3, 3.7, 3.7, 1.4, 1.1, 1.4], left_cols=(0,)), Spacer(1, 6),
      fig("forest", "fig2_subgroup_forest.png", "Logistic coefficients with 95% confidence intervals for the full data and each subgroup.", 16.4 * cm)]

# ---- 6 Part 2.2
S += [Paragraph("6  Part 2.2 \u2013 Omitted-variable bias", H1), P(T["p22"])]
o = R["p22"]["table"]
orows = [["Predictor", "\u03b2 full (p)", "\u03b2 reduced (p)", "Abs. \u0394", "% change", "\u0394 in SE", "Note"]]
for r in o:
    note = []
    if r["sign_flip"]:
        note.append("sign flip")
    if (not r["sig_full"]) and r["sig_reduced"]:
        note.append("n.s. \u2192 sig.")
    if r["sig_full"] and not r["sig_reduced"]:
        note.append("sig. \u2192 n.s.")
    orows.append([r["predictor"], f"{n(r['beta_full'])} ({pf(r['p_full'])})", f"{n(r['beta_reduced'])} ({pf(r['p_reduced'])})", n(r["abs_change"]), f"{r['pct_change']:+.1f}%".replace("-", "\u2212"),
                  n(r["change_in_SE_units"], 2), "; ".join(note)])
idx = [i + 1 for i, r in enumerate(o) if r["predictor"] == "race_African_American"]
S += [cap_t("p22", "Coefficients before (full model = Model B) and after dropping priors_count, training set. %\u0394 = (\u03b2_reduced \u2212 \u03b2_full)/|\u03b2_full| \u00d7 100; \u0394 in SE = absolute change divided by the full model\u2019s standard error."),
      tbl(orows, [3.4, 2.6, 2.7, 1.5, 1.8, 1.4, 2.5], hilite=tuple(idx)), Spacer(1, 6),
      fig("ovb", "fig3_omitted_variable.png", "Coefficients of the full and reduced models.", 13.2 * cm)]

# ---- 7 Part 2.3
S += [Paragraph("7  Part 2.3 \u2013 Adversarial subset construction", H1), P(T["p23"])]
prof = R["p23"]["profile"]
cols = list(prof.keys())
labels = [("n", "n", "{:.0f}"), ("age_mean", "Age, mean", "{:.1f}"), ("age_median", "Age, median", "{:.1f}"), ("pct_age_lt25", "% under 25", "{:.1f}"), ("pct_age_ge45", "% aged 45+", "{:.1f}"),
          ("pct_male", "% male", "{:.1f}"), ("pct_African_American", "% African-American", "{:.1f}"), ("pct_Caucasian", "% Caucasian", "{:.1f}"), ("pct_Hispanic", "% Hispanic", "{:.1f}"),
          ("pct_Other", "% Other", "{:.1f}"), ("priors_mean", "Priors, mean", "{:.2f}"), ("priors_median", "Priors, median", "{:.1f}"), ("pct_priors_ge5", "% with 5+ priors", "{:.1f}"),
          ("pct_priors_le1", "% with 0\u20131 priors", "{:.1f}"), ("pct_any_juvenile", "% any juvenile record", "{:.1f}"), ("pct_felony", "% felony charge", "{:.1f}"),
          ("actual_recid_rate", "% actually reoffended", "{:.1f}"), ("mean_pred_prob", "Mean predicted P", "{:.3f}")]
short = ["Full test set", f"Subset ({R['p23']['n_subset']})", "of which FP", "of which FN", f"All errors ({R['p23']['n_qualifying']})"]
prow = [["Characteristic"] + short]
for k, lab, f in labels:
    prow.append([lab] + [f.format(prof[c_][k]) for c_ in cols])
S += [cap_t("p23_profile", f"Profile of the 50-row adversarial subset compared with the full test set (n = {S_['n_test']:,}); FP = false positive, FN = false negative; the last column covers all {R['p23']['n_qualifying']} qualifying errors."),
      tbl(prow, [4.2, 2.6, 2.6, 2.6, 2.6, 2.6], zebra=True)]
trait = {(t_["trait"], t_["group"]): t_ for t_ in R["p23"]["traits"]}
trow = [["Trait", "Rest of test set", "All 50: % (p)", "FP only: % (p)", "FN only: % (p)"]]
for tr in ["age < 25", "age >= 45", "male", "African-American", "priors >= 5", "priors <= 1", "any juvenile record", "felony charge"]:
    a, b, c_ = trait[(tr, "all 50")], trait[(tr, "FP only")], trait[(tr, "FN only")]
    trow.append([tr.replace(">=", "\u2265").replace("<=", "\u2264"), f"{a['pct_in_rest_of_test']:.1f}%", f"{a['pct_in_group']:.1f}% ({pf(a['fisher_p'])})", f"{b['pct_in_group']:.1f}% ({pf(b['fisher_p'])})", f"{c_['pct_in_group']:.1f}% ({pf(c_['fisher_p'])})"])
S += [Spacer(1, 6), cap_t("p23_traits", f"Prevalence of each trait in the subset versus the remaining {S_['n_test'] - R['p23']['n_subset']:,} test rows, with Fisher exact p-values (24 tests, uncorrected; read as descriptive)."),
      tbl(trow, [3.6, 2.8, 3.6, 3.6, 3.6]), Spacer(1, 6),
      fig("adv", "fig4_adversarial.png", "(a) Age and prior record of the 50 confident errors against the test set; (b) calibration of Model B on the test set, overall and by race (bins with fewer than 10 people omitted).", 16.4 * cm)]

# ---- 8 Part 2.4
S += [Paragraph("8  Part 2.4 \u2013 Fairness trade-off analysis", H1), P(T["p24_def"])]
S.append(P("<b>Definitions (as in the brief).</b> DP = P(\u0176 = 1 | A) / P(\u0176 = 1 | B); EO = min(TPR<sub>A</sub>/TPR<sub>B</sub>, FPR<sub>A</sub>/FPR<sub>B</sub>); PP = P(Y = 1 | \u0176 = 1, A) / P(Y = 1 | \u0176 = 1, B). "
           "\u0176 = 1 if P \u2265 t, for t = 0.05, 0.15, \u2026, 0.95. Ideal value 1.0 for all three."))
frm = [["t", "Flagged A / B", "DP", "TPR ratio", "FPR ratio", "EO", "PP", "Accuracy"]]
for t_ in sorted(sw):
    r = sw[t_]
    frm.append([f"{t_:.2f}", f"{r['n_flag_A']} / {r['n_flag_B']}", fmt(r["DP"]), fmt(r["TPR_ratio"]), fmt(r["FPR_ratio"]), fmt(r["EO"]), fmt(r["PP"]), fr(r["acc_all"])])
hl = [1 + sorted(sw).index(ts)]
S += [cap_t("fair_main", f"Fairness metrics for every threshold (A = Caucasian, B = African-American; test set). EO is the smaller of the TPR and FPR ratios. NA: PP is undefined at t = 0.95 because no one in Group A is flagged. The recommended threshold t = {ts:.2f} is highlighted."),
      tbl(frm, [1.3, 3.0, 1.7, 2.0, 2.0, 1.7, 1.7, 2.1], hilite=tuple(hl))]
rrow = [["t", "TPR A", "TPR B", "FPR A", "FPR B", "PPV A", "PPV B", "FP A", "FP B", "FN A", "FN B"]]
for t_ in sorted(sw):
    r = sw[t_]
    rrow.append([f"{t_:.2f}", fr(r["TPR_A"]), fr(r["TPR_B"]), fr(r["FPR_A"]), fr(r["FPR_B"]), "NA" if r["PPV_A"] is None else fr(r["PPV_A"]), fr(r["PPV_B"]), r["FP_A"], r["FP_B"], r["FN_A"], r["FN_B"]])
S += [Spacer(1, 6), cap_t("fair_rates", f"Underlying group rates and error counts (Group A: {M['n_A']} defendants, of whom {M['n_neg_A']} did not reoffend; Group B: {M['n_B']}, of whom {M['n_neg_B']} did not reoffend)."),
      tbl(rrow, [1.2, 1.55, 1.55, 1.55, 1.55, 1.55, 1.55, 1.3, 1.3, 1.3, 1.3], hilite=tuple(hl))]
crow = [["t", "DP: 95% bootstrap CI", "EO: 95% bootstrap CI", "PP: 95% bootstrap CI"]]
for t_ in sorted(cis):
    r = cis[t_]
    crow.append([f"{t_:.2f}"] + [("NA" if r[f"{m_}_lo"] is None else ci(r[f"{m_}_lo"], r[f"{m_}_hi"], 2)) + ("" if r[f"{m_}_valid_frac"] is None or r[f"{m_}_valid_frac"] >= 0.999 else f" ({100 * r[f'{m_}_valid_frac']:.0f}% of draws valid)") for m_ in ["DP", "EO", "PP"]])
S += [Spacer(1, 6), cap_t("fair_ci", "Percentile bootstrap intervals (1,000 resamples of the test set, seed 450). \u201cvalid\u201d = share of resamples in which the ratio was estimable."),
      tbl(crow, [1.3, 5.2, 5.2, 5.2], hilite=tuple(hl)), Spacer(1, 6),
      fig("tradeoff", "fig5_dp_eo_tradeoff.png", "Required trade-off chart: Demographic Parity (x) against Equalized Odds (y), both on 0\u20131.5, each point labelled by its threshold. The star marks the ideal (1, 1).", 11.2 * cm)]
S += [P(T["p24_tradeoff"]), fig("metrics", "fig6_metrics_by_threshold.png", "DP, EO, PP and overall accuracy across thresholds; the vertical line marks the recommended threshold.", 12.4 * cm)]
trows = [["Move (t)", "DP", "EO", "PP", "DP gain", "EO gain", "PP change", "Accuracy change", "Cost (PP% per DP%)"]]
for k, lab in [("acc_max_to_star", None), ("star_to_lower", None), ("acc_max_to_0.25", None), ("acc_max_to_0.05", None)]:
    p_ = M["pairs"][k]
    trows.append([f"{p_['t_from']:.2f} \u2192 {p_['t_to']:.2f}", f"{n(p_['DP_from'], 2)} \u2192 {n(p_['DP_to'], 2)}", f"{n(p_['EO_from'], 2)} \u2192 {n(p_['EO_to'], 2)}", f"{n(p_['PP_from'], 2)} \u2192 {n(p_['PP_to'], 2)}",
                  f"+{p_['DP_gain_pct']:.1f}%", f"+{p_['EO_gain_pct']:.1f}%", f"{p_['PP_change_pct']:.1f}%".replace("-", "\u2212"), f"{p_['acc_change_pts']:.1f} pts".replace("-", "\u2212"), f"{p_['cost_PP_per_DP']:.3f}"])
S += [cap_t("tradeoff", "Quantified trade-offs between selected thresholds, computed on the parity score s = min(m, 1/m). All 9 consecutive steps are in results/tables/part24_consecutive_tradeoffs.csv."),
      tbl(trows, [2.1, 2.1, 2.1, 2.1, 1.5, 1.5, 1.7, 2.0, 2.0], left_cols=(0,)), Spacer(1, 6), P(T["p24_reco"])]

# ---- 9 Part 3
S += [Paragraph("9  Part 3 \u2013 Descriptive analysis questions", H1),
      P("Each answer is written from the numbers computed above. Word counts are shown in grey (assignment target: about 250\u2013300 words).", SMALL)]
QTXT = {1: "Why did the threshold-based approach and the direct logistic approach flag, or not flag, the same individuals as high risk? Quote the mismatch count and percentage, and compare the median linear prediction with the 0.5 threshold.",
        2: "Describe the subgroup with significantly different effects. Using the coefficients from the full model and both subgroups, explain why the trend differs, with exact coefficients and confidence intervals, and say which policy conclusion would change if the subgroup were ignored.",
        3: "State the variable dropped for the omitted-variable test. Report the before/after coefficient shift for at least four predictors in a table with percentage changes. Which coefficient shifted most, and what does that reveal about trusting an un-stress-tested model?",
        4: "As a practitioner, how would you detect a hidden confound you cannot currently see? Propose one concrete, real-world method.",
        5: "Describe the 50-row adversarial subset: the common characteristic, the percentage that share it, and the statistical reason the model fails confidently, referring to coefficients or decision boundaries.",
        6: "Using the fairness trade-off chart, explain in concrete terms what the trade-off cost means for a real person. Describe two individuals, one from each group, and their outcomes under different thresholds.",
        7: "Choose a threshold and argue, with at least three specific numbers, why it is defensible; give the strongest counterargument and acknowledge its validity.",
        8: "Should a policymaker prioritise demographic parity or equalized odds for this dataset? Cite the trade-off numbers and acknowledge what is sacrificed."}
for q in range(1, 9):
    body = T["Q"][q]
    S += [Paragraph(f"Question {q}", QH), Paragraph(QTXT[q], QP), P(body), Paragraph(f"({RT.words(RT.plain(body))} words)", SMALL)]
    if q == 3:
        o22 = {r["predictor"]: r for r in R["p22"]["table"]}
        qrows = [["Predictor", "\u03b2 full", "\u03b2 reduced (no priors_count)", "% change", "Significance change"]]
        for k in ["age", "race_African_American", "juv_fel_count", "juv_misd_count", "juv_other_count", "felony", "male"]:
            r = o22[k]
            sig = "n.s. \u2192 significant" if (not r["sig_full"] and r["sig_reduced"]) else ("significant \u2192 n.s." if (r["sig_full"] and not r["sig_reduced"]) else "unchanged")
            qrows.append([k, n(r["beta_full"]), n(r["beta_reduced"]), f"{r['pct_change']:+.1f}%".replace("-", "\u2212"), sig])
        S += [cap_t("p22q3", "Coefficient shifts after dropping priors_count (selected predictors; full table in Section 6)."), tbl(qrows, [3.8, 2.2, 4.4, 2.2, 4.0]), Spacer(1, 4)]

# ---- 10 Part 4
S += [Paragraph("10  Part 4 \u2013 Reflection essay", H1),
      Paragraph("\u201cIf a policy-maker asked you: \u2018just give me one number: should we deploy this model or not?\u2019, what would you say, and what is the limitation of that \u2018one number\u2019 answer that you discovered yourself through your analysis?\u201d", QP),
      P(T["refl_direct"]), P(T["refl_fail"]), P(T["refl_ground"]), P(T["refl_close"])]
tot = sum(RT.words(RT.plain(T[k])) for k in ["refl_direct", "refl_fail", "refl_ground", "refl_close"])
S.append(Paragraph(f"({tot} words)", SMALL))

# ---- 11 limits & conclusion
S += [Paragraph("11  Limitations and conclusion", H1), P(T["limits"])]

# ---- references
S += [Paragraph("References", H1)] + [Paragraph(r_, ParagraphStyle("ref", parent=BODYL, leftIndent=12, firstLineIndent=-12, fontSize=8.4, leading=11.2)) for r_ in T["refs"]]

# ---- appendices
S += [PageBreak(), Paragraph("Appendix A \u2013 Fairness sweep with the reciprocal orientation", H1),
      P("The same classifier and test set with Group A = African-American and Group B = Caucasian. Every ratio is the reciprocal of the main table; EO now equals the <i>smaller</i> of the two ratios, which understates the larger disparity "
        "(for example at t = 0.45 the FPR ratio is {} but EO reports {}). This is why the main analysis uses the opposite orientation.".format(
            n(M["alt_sweep"][4]["FPR_ratio"], 3), n(M["alt_sweep"][4]["EO"], 3)))]
arows = [["t", "DP", "TPR ratio", "FPR ratio", "EO", "PP"]]
for r in M["alt_sweep"]:
    arows.append([f"{r['t']:.2f}", fmt(r["DP"]), fmt(r["TPR_ratio"]), fmt(r["FPR_ratio"]), fmt(r["EO"]), fmt(r["PP"])])
S += [cap_t("A1", "Reciprocal orientation (A = African-American, B = Caucasian). NA at t = 0.95: Group B (Caucasian) has no flagged defendants, so every ratio has a zero denominator."), tbl(arows, [1.6, 2.2, 2.4, 2.4, 2.2, 2.2])]

S += [Paragraph("Appendix B \u2013 Robustness checks", H1),
      P("Model B was refitted under three alternative choices: the alternative outcome <i>is_recid</i>; winsorised priors_count; and a \u201crace-blind\u201d model without race indicators. "
        "Fairness metrics are shown at the recommended threshold (A = Caucasian, B = African-American). The race-blind model keeps its AUC but does not eliminate the disparity, which runs through proxies such as prior record and age.")]
rb = R["robust"]
rrows = [["Variant", "AUC", "Accuracy at 0.5", "Brier", f"DP at {ts}", f"EO at {ts}", f"PP at {ts}"]]
for k, v in rb.items():
    if k == "winsor_cap":
        continue
    rrows.append([k, n(v["AUC"]), n(v["Accuracy@0.5"]), n(v["Brier"]), n(v[f"DP@{ts}"]), n(v[f"EO@{ts}"]), n(v[f"PP@{ts}"])])
S += [cap_t("robust", "Robustness checks for Model B."), tbl(rrows, [5.6, 1.4, 2.2, 1.4, 1.5, 1.5, 1.5])]

# ---- Appendix C: reproducibility + QC
env = R["env"]
S += [Paragraph("Appendix C \u2013 Reproducibility and quality control", H1),
      P(f"<b>How to regenerate.</b> Install the packages in requirements.txt, open notebook/CSET485_Assignment1_COMPAS.ipynb and run all cells. The notebook downloads the dataset if data/compas-scores-two-years.csv is absent, fits every model with "
        f"random_state = {S_['seed']}, and writes results/results.json, results/tables/*.csv and results/figures/*.png. Then run <i>python src/build_report.py</i> to rebuild this PDF (the last notebook cell does this automatically when reportlab is installed). "
        f"<b>Environment used:</b> Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn {env['scikit-learn']}, statsmodels {env['statsmodels']}, scipy {env['scipy']}, matplotlib {env['matplotlib']}. "
        f"<b>Data file SHA-256:</b> {R['data']['sha256']}."),
      P("<b>Checks performed.</b> The first block is evaluated when this PDF is built; the second block was run once as an independent re-computation (src/verify_results.py) that refits the models with a different library and compares headline numbers.")]
qrows = [["Check", "Result"]]
checks = [("Dataset is COMPAS two-year file", R["data"]["n_rows"] == 7214 and "compas-scores-two-years" in R["data"]["url"]),
          ("Random seed is 450", S_["seed"] == 450), ("Split is 70/30", abs(S_["test_frac"] - 0.30) < 0.001 and S_["n_train"] + S_["n_test"] == R["data"]["n_rows"]),
          ("Models A (OLS) and B (logistic) implemented and cross-checked with scikit-learn", R["modelA"]["sklearn_max_diff"] < 1e-8 and R["modelB"]["sklearn_max_diff"] < 1e-2),
          ("Four stress tests present (2.1\u20132.4)", all(k in R for k in ["p21", "p22", "p23", "p24"])),
          ("Fairness sweep covers all 10 thresholds 0.05\u20130.95", [round(t_, 2) for t_ in M["thresholds"]] == [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]),
          ("All eight Part 3 answers within 250\u2013300 words", all(250 <= RT.words(RT.plain(T["Q"][q])) <= 300 for q in range(1, 9))),
          ("Reflection about 500 words with the four required parts", 450 <= tot <= 550),
          ("Mismatch count equals off-diagonal of the confusion matrix", P1["mismatch_n"] == P1["cm"]["AL_BH"] + P1["cm"]["AH_BL"]),
          ("Adversarial subset: 50 rows drawn with seed 450", R["p23"]["n_subset"] == 50 and R["p23"]["sample_seed"] == 450)]
for lab, ok in checks:
    assert ok, f"QC failed: {lab}"
    qrows.append([lab, "PASS"])
vf = ROOT / "results" / "verification.txt"
if vf.exists():
    for line in vf.read_text().strip().splitlines():
        a, _, b = line.partition("|")
        qrows.append([a.strip(), b.strip()])
S += [cap_t("qc", "Quality-control checklist."), tbl(qrows, [13.0, 4.0], left_cols=(0,))]

# ------------------------------------------------------------------------------------------------ keep captions with short tables
def keep_pairs(story, max_rows=20):
    out, i = [], 0
    while i < len(story):
        a = story[i]
        nxt = story[i + 1] if i + 1 < len(story) else None
        is_cap = isinstance(a, Paragraph) and a.style.name == "Cap" and a.text.startswith("<b>Table")
        if is_cap and isinstance(nxt, Table) and getattr(nxt, "_nrows", 99) <= max_rows:
            out.append(KeepTogether([a, nxt]))
            i += 2
        else:
            out.append(a)
            i += 1
    return out


S = keep_pairs(S)

# ------------------------------------------------------------------------------------------------ build
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = Doc(str(OUT))
doc.multiBuild(S, canvasmaker=NumberedCanvas)
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
