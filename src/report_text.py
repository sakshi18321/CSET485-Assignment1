"""
Narrative text for the CSET485 Assignment 1 report.

Every number below is read from ``results/results.json`` (written by the notebook), so the PDF text cannot
drift from the computed results.  A few ``assert`` statements guard the qualitative claims: if the notebook is
re-run on different data and a claim stops being true, the build fails instead of printing a false sentence.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    return json.loads((ROOT / "results" / "results.json").read_text())


# ----------------------------------------------------------------------------- formatting helpers
def n(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "\u2212")


def pv(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def pct(x, d=1):          # x already in percent
    return f"{x:.{d}f}%"


def fr(x, d=1):           # x a fraction
    return f"{100 * x:.{d}f}%"


def ci(lo, hi, d=3):
    return f"[{n(lo, d)}, {n(hi, d)}]"


def words(s):
    """Whitespace-separated tokens, the same convention as Word / Google Docs word counts."""
    return len(s.split())


def build(R):
    P1, M, S = R["part1"], R["p24"], R["split"]
    A, B = R["modelA"], R["modelB"]
    p21, p22, p23 = R["p21"], R["p22"], R["p23"]
    sx, rc = p21["primary"], p21["race"]
    sw = {round(r["t"], 2): r for r in M["sweep"]}
    cis = {round(r["t"], 2): r for r in M["ci"]}
    ts = M["t_star"]
    star, sci = sw[ts], cis[ts]
    tb = M["t_best_acc"]
    best = sw[tb]
    pairs = M["pairs"]
    T = {}

    # ---------------------------------------------------------------- handy extracted values
    coefA, coefB = A["coef"], B["coef"]
    sxt = {r["predictor"]: r for r in sx["table"]}
    rct = {r["predictor"]: r for r in rc["table"]}
    o22 = {r["predictor"]: r for r in p22["table"]}
    pri = sxt["priors_count"]
    lg = sx["log_check"]
    cal = sx["pooled_calibration"]
    perf = sx["group_vs_pooled_test"]
    ext = p23["ext_model"]
    prof = p23["profile"]
    sub_col, fp_col, fn_col, test_col, allq_col = (list(prof.keys())[i] for i in (1, 2, 3, 0, 4))
    sub_p, fp_p, fn_p, te_p, aq_p = prof[sub_col], prof[fp_col], prof[fn_col], prof[test_col], prof[allq_col]
    race_perf = rc["group_vs_pooled_test"]
    ill = {(d["profile"][:12], d["race"]): d["p"] for d in M["illustration"]}
    ill30_aa = [d for d in M["illustration"] if d["profile"].startswith("30") and d["race"] == "African-American"][0]["p"]
    ill30_c = [d for d in M["illustration"] if d["profile"].startswith("30") and d["race"] == "Caucasian"][0]["p"]

    # ---------------------------------------------------------------- guards on qualitative claims
    assert P1["AhBl"] > P1["AlBh"]
    assert coefA["race_African_American"]["p"] < 0.001 and coefB["race_African_American"]["p"] > 0.05
    assert pri["B_coef"] > pri["A_coef"] and pri["diff_p_bonferroni"] < 0.05
    assert p21["primary"]["log_check"]["diff_p"] > 0.05
    assert o22["race_African_American"]["p_full"] > 0.05 > o22["race_African_American"]["p_reduced"]
    assert p23["n_qualifying"] > 50 and p23["subset_n_fp"] > 0 and p23["subset_n_fn"] > 0
    assert fp_p["pct_priors_ge5"] == 100.0 and fn_p["pct_priors_ge5"] == 0.0
    assert sub_p["pct_age_lt25"] == 0.0 and fn_p["pct_any_juvenile"] == 0.0
    assert R["p21_scan"]["sign_flips_with_both_sig"] == 0 and all(f["p_bonferroni"] > 0.05 for f in R["p21_scan"]["sign_flips"])
    assert M["corr_DP_EO"] > 0.95 and M["n_steps_DP_up_EO_down"] == 0
    assert star["DP"] < 1 and star["EO"] < 1 and star["FPR_B"] > star["FPR_A"]

    trait = {(t["trait"], t["group"]): t for t in p23["traits"]}
    va, vb = P1["vs_actual"]["Model A flag (>= median)"], P1["vs_actual"]["Model B flag (p >= 0.5)"]
    cal_r = rc["pooled_calibration"]
    # derived numbers used in several places
    fp_tot = {t: sw[t]["FP_A"] + sw[t]["FP_B"] for t in sw}
    tol = M["tol_to_threshold"]
    assert tol["0.01"] == tol["0.02"] == tol["0.03"] == tol["0.05"] == ts and tol["0.07"] == tol["0.1"] != ts
    assert 8 < sw[0.25]["FPR_A"] / sw[0.55]["FPR_A"] < 11                      # "almost tenfold" in Q6
    assert sci["PP_lo"] < 1 < sci["PP_hi"] and sci["DP_hi"] < 1 and sci["EO_hi"] < 1
    pp_near = [t for t in sorted(sw) if sw[t]["PP"] is not None and abs(sw[t]["PP"] - 1) <= 0.03]
    pp_near_txt = ", ".join(f"{t:.2f}" for t in pp_near[:-1]) + f" and {pp_near[-1]:.2f}"
    pp_near_eo = max(sw[t]["EO"] for t in pp_near)
    assert 0.25 < min(M['base_rate_A'], M['base_rate_B']) and ts == 0.45
    mm = P1["mismatch_n"]
    ex_tot = p23["expected_fp_if_calibrated"] + p23["expected_fn_if_calibrated"]

    # ================================================================================== SHORT SECTIONS
    T["abstract"] = (
        f"This report stress-tests risk models built on the COMPAS two-year recidivism file (n = {R['data']['n_rows']:,}; "
        f"seed {S['seed']}; {S['n_train']:,} training and {S['n_test']:,} test rows). A linear model of the COMPAS decile score (Model A) and a logistic model of "
        f"actual two-year recidivism (Model B; test accuracy {fr(B['accuracy'])}, AUC {n(B['auc'])}) disagree on {mm} test defendants ({pct(P1['mismatch_pct'], 2)}), "
        f"almost all of them flagged by Model A only. Four stress tests follow. (1) No genuine sign reversal exists; the clearest subgroup difference is a stronger "
        f"prior-record effect for women ({n(pri['B_coef'])} versus {n(pri['A_coef'])} for men), which shrinks once priors are log-transformed. (2) Omitting priors_count turns the race "
        f"coefficient from {n(o22['race_African_American']['beta_full'])} (not significant) into {n(o22['race_African_American']['beta_reduced'])} (p {pv(o22['race_African_American']['p_reduced'])}). "
        f"(3) {p23['n_qualifying']} confident errors ({pct(p23['pct_of_test'], 2)} of the test set) are almost exactly as many as the model's own probabilities predict. "
        f"(4) Across thresholds 0.05 to 0.95, demographic parity and equalized odds rise and fall together (r = {n(M['corr_DP_EO'])}); the real cost of parity is accuracy and "
        f"predictive parity. I recommend conditional, advisory-only use at t = {ts:.2f} (DP {n(star['DP'])}, EO {n(star['EO'])}, PP {n(star['PP'])}, accuracy {fr(star['acc_all'])}), "
        f"with the limitations documented below."
    )

    T["intro"] = (
        "Predictive models used in high-stakes decisions can look statistically sound while concealing bias, hidden confounding and fairness trade-offs. "
        "This assignment asks for a model to be built, deliberately attacked, and then defended in writing using only numbers the student computed. "
        "The objectives are: (i) fit a linear (Model A) and a logistic (Model B) model on the assigned dataset and compare the high-risk labels they produce; "
        "(ii) run four stress tests: a Simpson's-paradox hunt, an omitted-variable experiment, an adversarial subset of confident errors, and a fairness-threshold sweep; "
        "(iii) answer eight analytical questions; and (iv) reflect on whether a policy-maker can responsibly be given \u201cjust one number\u201d. "
        "My roll number (0450) ends in 0, so the assigned dataset is the COMPAS Recidivism Risk Score file, and the random seed and split are 450 and 70/30. "
        "All results come from one notebook that regenerates every table, figure and number in this document."
    )

    T["dataset"] = (
        f"The data are ProPublica\u2019s <i>compas-scores-two-years.csv</i> ({R['data']['n_rows']:,} defendants, {R['data']['n_cols']} columns) from Broward County, Florida, "
        f"downloaded from the URL in the assignment (SHA-256 recorded in Appendix C). Defendant IDs are unique; <i>decile_score</i> ranges from {R['data']['decile_min']} to {R['data']['decile_max']}. "
        f"<b>Target-variable check.</b> The brief names a binary outcome <i>is_recidivist</i>, but no such column exists. Two candidates do: <i>two_year_recid</i> and <i>is_recid</i>, which disagree on "
        f"{R['target']['n_is_recid1_but_two_year0']} people who reoffended after the two-year window. I therefore define <i>is_recidivist := two_year_recid</i> because it alone has a fixed two-year follow-up; "
        f"its base rate is {fr(R['target']['base_rate_two_year'])} (versus {fr(R['target']['base_rate_is_recid'])} for <i>is_recid</i>). A sensitivity check with <i>is_recid</i> (Appendix B) changes AUC by only "
        f"{n(R['robust']['Target = is_recid']['AUC'] - R['robust']['Main model B (target two_year_recid)']['AUC'])}. "
        f"Features are age, priors_count, three juvenile-record counts, sex, charge degree (felony/misdemeanour) and race. The recidivism rate differs by group "
        f"([[tbl:dataset]]), as do prior records and age, which matters for every fairness result that follows."
    )

    T["preproc_intro"] = (
        "Each preprocessing decision is documented below, as the assignment requires. No outcome-derived or COMPAS-output column is used as a predictor, and every fitted "
        "transformation (the scaler) is estimated on the training set only."
    )

    T["part1"] = (
        f"<b>Models.</b> Model A is ordinary least squares on <i>decile_score</i>; Model B is unpenalised logistic regression on <i>is_recidivist</i>. Both use the same ten features and the same split, and "
        f"were cross-checked against scikit-learn (maximum prediction difference {A['sklearn_max_diff']:.1e} for A and {B['sklearn_max_diff']:.1e} for B). "
        f"On the test set Model A reaches R\u00b2 = {n(A['r2_test'])} (RMSE {n(A['rmse_test'])} deciles, MAE {n(A['mae_test'])}); its predictions range from {n(A['pred_min'], 2)} to {n(A['pred_max'], 2)}, "
        f"i.e. outside the valid 1\u201310 scale, a known weakness of linear models. Model B reaches accuracy {fr(B['accuracy'])}, AUC {n(B['auc'])}, precision {n(B['precision'])} and recall {n(B['recall'])} at the 0.5 threshold; "
        f"the majority-class baseline is {fr(S['majority_baseline_acc_test'])}, so the model adds {100 * (B['accuracy'] - S['majority_baseline_acc_test']):.1f} percentage points. "
        f"<b>Threshold comparison.</b> Model A\u2019s test median prediction is {n(P1['median_pred_A'])}; flagging predictions \u2265 that median labels {P1['n_flag_A']:,} defendants ({pct(100 * P1['flag_rate_A'], 2)}) high risk. "
        f"Flagging Model B probabilities \u2265 0.5 labels {P1['n_flag_B']:,} ({pct(100 * P1['flag_rate_B'], 2)}). The two labellings disagree on <b>{mm} defendants ({pct(P1['mismatch_pct'], 2)})</b>: "
        f"{P1['AhBl']} are high risk under A only and {P1['AlBh']} under B only. [[tbl:flags]] distinguishes this <i>agreement</i> from <i>accuracy against actual outcomes</i>. "
        f"The reasons for the disagreement are analysed in the answer to Question 1."
    )

    T["p21"] = (
        f"<b>Design.</b> Inference models were fitted on all {sx['n_full']:,} rows (full) and on each subgroup with identical features (the split variable omitted). Differences are tested with "
        f"z = (\u03b2<sub>A</sub> \u2212 \u03b2<sub>B</sub>)/\u221a(se<sub>A</sub>\u00b2 + se<sub>B</sub>\u00b2). Because I was searching for a reversal, I first scanned three demographic splits (race, sex, age \u2264 30 vs &gt; 30) "
        f"over every predictor: {R['p21_scan']['n_tests']} comparisons, so the Bonferroni threshold is p &lt; {R['p21_scan']['bonferroni_alpha']:.4f}. "
        f"<b>Finding 1: no genuine reversal.</b> {R['p21_scan']['n_sign_flips']} of the {R['p21_scan']['n_tests']} coefficients change sign between subgroups, but in none of them are both subgroup coefficients significantly non-zero "
        f"({R['p21_scan']['sign_flips_with_both_sig']} of {R['p21_scan']['n_sign_flips']}), and none is significant after correction; they are sampling noise around zero. For example, the race coefficient is "
        f"{n(sx['race_coef_by_sex']['male'][0])} {ci(*sx['race_coef_by_sex']['male'][1:3])} for men and {n(sx['race_coef_by_sex']['female'][0])} {ci(*sx['race_coef_by_sex']['female'][1:3])} for women "
        f"(difference p = {sx['race_coef_by_sex']['diff_p']:.3f}, Bonferroni p = {sx['race_coef_by_sex']['diff_p_bonferroni']:.2f}); I report it as an exploratory observation, not a finding. "
        f"<b>Finding 2: a significantly stronger effect.</b> The priors_count effect is {n(pri['B_coef'])} {ci(pri['B_lo'], pri['B_hi'])} for women (B, n = {sx['n_B']:,}) versus {n(pri['A_coef'])} {ci(pri['A_lo'], pri['A_hi'])} for men "
        f"(A, n = {sx['n_A']:,}); full data {n(pri['full_coef'])}. The intervals do not overlap (z = {n(abs(pri['diff_z']), 2)}, p = {pri['diff_p']:.4f}; Bonferroni p = {pri['diff_p_bonferroni']:.3f}). "
        f"It is the only sex-split difference that survives correction (age-split differences are partly mechanical because age is both the splitter and a predictor). "
        f"<b>Diagnostic.</b> Entering priors as log(1 + priors) shrinks the gap to {n(lg['male_beta'])} (men) versus {n(lg['female_beta'])} (women), p = {lg['diff_p']:.3f}, so much of the heterogeneity reflects "
        f"functional form: women have fewer priors (mean {n(sx['priors_by_sex']['Female']['mean'], 2)} versus {n(sx['priors_by_sex']['Male']['mean'], 2)}). "
        f"The race split, which matters most for Section 2.4, yields priors_count coefficients of {n(rct['priors_count']['A_coef'])} (African-American) versus {n(rct['priors_count']['B_coef'])} (Caucasian), "
        f"nominal p = {rct['priors_count']['diff_p']:.3f} but Bonferroni p = {rct['priors_count']['diff_p_bonferroni']:.2f}: suggestive, not established ([[tbl:p21_race]])."
    )

    T["p22"] = (
        f"<b>Choice of confounder: priors_count.</b> It is the strongest predictor of recidivism (full-model \u03b2 = {n(p22['priors_coef_full'])}, training-set correlation with the outcome r = {n(p22['corr_priors_outcome_train'], 2)}) and is correlated with the other "
        f"predictors, including race_African_American (r = {n(p22['corr_with_priors']['race_African_American'], 2)}), juv_misd_count (r = {n(p22['corr_with_priors']['juv_misd_count'], 2)}) and age (r = {n(p22['corr_with_priors']['age'], 2)}), "
        f"so removing it should push its explanatory power onto them. It is also shaped by policing intensity, which is why it is a contestable variable. The full model is Model B (training set); the reduced model drops only priors_count. "
        f"<b>Formula:</b> %\u0394 = (\u03b2<sub>reduced</sub> \u2212 \u03b2<sub>full</sub>) / |\u03b2<sub>full</sub>| \u00d7 100; because this explodes when \u03b2<sub>full</sub> \u2248 0, [[tbl:p22]] also reports the absolute change and the change in units of the full model\u2019s standard error. "
        f"<b>Results.</b> The largest percentage change is juv_misd_count ({n(o22['juv_misd_count']['beta_full'])} \u2192 {n(o22['juv_misd_count']['beta_reduced'])}, {o22['juv_misd_count']['pct_change']:+.0f}%, a sign flip), but its baseline is almost zero, so this is a numerical artefact. "
        f"The largest absolute and standardised shift is age ({n(o22['age']['beta_full'])} \u2192 {n(o22['age']['beta_reduced'])}, {o22['age']['pct_change']:+.1f}%, {n(o22['age']['change_in_SE_units'], 1)} standard errors). "
        f"The most consequential is race_African_American: {n(o22['race_African_American']['beta_full'])} (p {pv(o22['race_African_American']['p_full'])}) \u2192 {n(o22['race_African_American']['beta_reduced'])} (p {pv(o22['race_African_American']['p_reduced'])}), "
        f"{o22['race_African_American']['pct_change']:+.1f}%. { {1:'One',2:'Two',3:'Three',4:'Four',5:'Five'}.get(p22['n_sig_gain'], str(p22['n_sig_gain'])) } coefficients (the three juvenile counts and race_African_American) turn from non-significant to significant and none loses significance. "
        f"Test AUC falls from {n(p22['perf_full']['auc'])} to {n(p22['perf_reduced']['auc'])}. <b>Caveats.</b> Logistic coefficients are non-collapsible (they change when covariates are dropped even without confounding), and priors may be a mediator "
        f"rather than a confounder; the table shows how fragile coefficient interpretations are, not the causal effect of any variable."
    )

    T["p23"] = (
        f"<b>Extraction.</b> On the {S['n_test']:,} test rows there are {p23['n_fp']} false positives (P \u2265 0.75, actual 0) and {p23['n_fn']} false negatives (P \u2264 0.25, actual 1): <b>{p23['n_qualifying']} confident errors, "
        f"{pct(p23['pct_of_test'], 2)} of the test set</b>. Because more than 50 qualify, 50 were drawn at random (<i>random_state</i> = {p23['sample_seed']}): {p23['subset_n_fp']} false positives and {p23['subset_n_fn']} false negatives. "
        f"<b>Profile.</b> Compared with the full test set, the subset contains <b>no one under 25</b> (0% versus {pct(te_p['pct_age_lt25'])}), is much older (median age {sub_p['age_median']:.1f} versus {te_p['age_median']:.0f}; "
        f"{pct(sub_p['pct_age_ge45'], 0)} aged 45+ versus {pct(te_p['pct_age_ge45'])}), and has a different gender mix ({pct(sub_p['pct_male'], 0)} male versus {pct(te_p['pct_male'])}). "
        f"It is two mirror-image clusters rather than one: all {p23['subset_n_fp']} false positives have \u2265 5 priors (median {fp_p['priors_median']:.0f}; {pct(fp_p['pct_African_American'], 0)} African-American, {pct(fp_p['pct_male'], 0)} male), "
        f"whereas the {p23['subset_n_fn']} false negatives have none with \u2265 5 priors ({pct(fn_p['pct_priors_le1'], 0)} have at most one, {pct(fn_p['pct_age_ge45'], 0)} are 45 or older, {pct(fn_p['pct_African_American'], 0)} African-American). "
        f"<b>Why the model fails confidently.</b> (a) <i>Calibrated uncertainty:</i> among the {p23['n_conf_pos']} test defendants with P \u2265 0.75 the observed recidivism rate is {fr(p23['obs_pos_rate_when_p_ge_75'])} (mean predicted {fr(p23['mean_p_when_p_ge_75'])}), "
        f"and among the {p23['n_conf_neg']} with P \u2264 0.25 it is {fr(p23['obs_pos_rate_when_p_le_25'])} (predicted {fr(p23['mean_p_when_p_le_25'])}). A calibrated model should therefore make about {p23['expected_fp_if_calibrated']:.1f} confident false positives "
        f"and {p23['expected_fn_if_calibrated']:.1f} confident false negatives ({ex_tot:.1f} in total); the observed {p23['n_fp']} and {p23['n_fn']} ({p23['n_qualifying']}) are close, so most of these errors are irreducible outcome noise. "
        f"(b) <i>Ignored non-linearity:</i> adding squared and interaction terms improves fit significantly (LR \u03c7\u00b2 = {ext['lr_stat']:.1f}, df = {ext['df']}, p &lt; 0.001) with priors\u00b2 = {n(ext['coef']['priors_sq']['coef'])} (diminishing returns) and age\u00b2 = {n(ext['coef']['age_sq']['coef'])} "
        f"(older defendants are not as safe as a straight line implies), but test AUC moves only from {n(ext['auc_base'])} to {n(ext['auc_ext'])} and the count of confident errors does not fall ({ext['n_conf_errors_base']} \u2192 {ext['n_conf_errors_ext']}). "
        f"Misspecification contributes, yet it does not explain most of the failures."
    )

    # ------------------------------------------------------------------ 2.4 text
    p1_, p2_, p3_ = pairs["acc_max_to_star"], pairs["star_to_lower"], pairs["acc_max_to_0.25"]
    T["p24_def"] = (
        f"<b>Groups.</b> Sensitive variable: race. Group A = Caucasian (n = {M['n_A']}), Group B = African-American (n = {M['n_B']}) in the test set (other race groups are excluded here). "
        f"Test-set base rates are {fr(M['base_rate_A'])} (A) and {fr(M['base_rate_B'])} (B); mean Model B probabilities are {n(M['mean_p_A'], 2)} and {n(M['mean_p_B'], 2)}. "
        f"<b>Orientation.</b> The EO formula takes the <i>minimum</i> of two ratios. With the more-flagged group in the denominator both ratios are \u2264 1 and the minimum reports the <i>larger</i> disparity; with the reverse orientation the minimum would report the smaller one and "
        f"understate the problem. This orientation also keeps all points inside the required 0\u20131.5 axes. The reciprocal (A = African-American) table is in Appendix A. "
        f"<b>Zero denominators.</b> A ratio is NA if its denominator is zero or a needed rate is undefined. This occurs once, at t = 0.95: nobody in Group A is flagged, so PP is NA (DP and EO are computed as 0 because only the numerator is zero). "
        f"At that threshold only {sw[0.95]['n_flag_B']} Group B members are flagged, so those values carry almost no information."
    )

    T["p24_tradeoff"] = (
        f"<b>Parity score.</b> To compare gains and losses across metrics that can lie on either side of 1, each ratio m is converted to s = min(m, 1/m) (1 = perfect parity); percentages below are relative changes in s. "
        f"<b>DP and EO do not trade off here.</b> Across the sweep they are almost perfectly correlated (r = {n(M['corr_DP_EO'])}), and no one-step reduction in threshold improves DP while worsening EO (0 of {len(sw) - 1} steps), "
        f"because EO\u2019s binding term is the false-positive-rate ratio, which tracks the flag-rate ratio. The assignment\u2019s template therefore cannot be completed with EO as the loser; the costs of parity land on predictive parity and accuracy. "
        f"<b>Quantified trade-off.</b> If you improve DP from {n(p1_['DP_from'])} (t = {p1_['t_from']:.2f}, the accuracy-maximising threshold) to {n(p1_['DP_to'])} (t = {p1_['t_to']:.2f}), a {p1_['DP_gain_pct']:.1f}% gain, EO also rises from {n(p1_['EO_from'])} to {n(p1_['EO_to'])} "
        f"(+{p1_['EO_gain_pct']:.1f}%), but PP moves away from parity from {n(p1_['PP_from'])} to {n(p1_['PP_to'])} (a {abs(p1_['PP_change_pct']):.1f}% loss) and accuracy falls {abs(p1_['acc_change_pts']):.2f} points. The trade-off cost is {abs(p1_['PP_change_pct']):.1f}/{p1_['DP_gain_pct']:.1f} = <b>{p1_['cost_PP_per_DP']:.2f}</b> "
        f"percent of PP per percent of DP. One step further (t = {p2_['t_from']:.2f} \u2192 {p2_['t_to']:.2f}) buys DP +{p2_['DP_gain_pct']:.1f}% and EO +{p2_['EO_gain_pct']:.1f}% for a PP loss of only {abs(p2_['PP_change_pct']):.1f}% (cost {p2_['cost_PP_per_DP']:.2f}) but costs "
        f"{abs(p2_['acc_change_pts']):.1f} accuracy points ({fr(p2_['acc_from'])} \u2192 {fr(p2_['acc_to'])}) and raises the number of wrongly flagged non-reoffenders in the two groups from {fp_tot[ts]} to {fp_tot[round(ts - 0.1, 2)]}. "
        f"Pushing to t = 0.25 doubles DP (+{p3_['DP_gain_pct']:.0f}%) at an accuracy cost of {abs(p3_['acc_change_pts']):.1f} points. The ratios reach 1.0 only at t = 0.05, where everyone is flagged (accuracy {fr(sw[0.05]['acc_all'])}, below the {fr(S['majority_baseline_acc_test'])} majority baseline): "
        f"perfect parity is available only by abandoning prediction. <b>Key insight:</b> no threshold satisfies all three metrics. PP is within 3% of 1 only at t = {pp_near_txt}, and at those thresholds EO is at most {n(pp_near_eo, 2)}."
    )

    T["p24_reco"] = (
        f"<b>Decision rule and recommendation.</b> Among thresholds whose accuracy is within 2 percentage points of the best ({fr(best['acc_all'])} at t = {tb:.2f}), I choose the one with the highest worst-case parity score, min(s<sub>DP</sub>, s<sub>EO</sub>, s<sub>PP</sub>). "
        f"The rule selects <b>t = {ts:.2f}</b>, and the choice is unchanged for tolerances of 1 to 5 points but moves to t = {tol['0.07']:.2f} at 7 to 10 points, so the 2-point tolerance is a stated value judgment, not a data-driven fact. "
        f"At t = {ts:.2f}: accuracy {fr(star['acc_all'])}; DP {n(star['DP'])} (95% bootstrap CI {ci(sci['DP_lo'], sci['DP_hi'])}); EO {n(star['EO'])} {ci(sci['EO_lo'], sci['EO_hi'])}; PP {n(star['PP'])} {ci(sci['PP_lo'], sci['PP_hi'])}. "
        f"PP is statistically indistinguishable from parity, but DP and EO are not: the false-positive rate is {fr(star['FPR_B'])} for African-American versus {fr(star['FPR_A'])} for Caucasian defendants, and "
        f"{M['fp_B_at_star']} versus {M['fp_A_at_star']} non-reoffenders are wrongly flagged. I recommend this threshold only for advisory use, paired with the monitoring described in Part 4."
    )

    # ================================================================================== PART 3
    Q = {}
    # ---- Q1
    Q[1] = (
        f"The two rules disagreed on {mm} of {P1['n_test']:,} test defendants ({pct(P1['mismatch_pct'], 2)}) and agreed on {P1['agree_n']:,} ({pct(P1['agree_pct'], 2)}). "
        f"The disagreement is nearly one-directional: {P1['AhBl']} people were high risk under Model A but low risk under Model B, against only {P1['AlBh']} the other way. Three mechanisms explain it. "
        f"First, the cut-offs mean different things. Model A flags predictions at or above the median of its own output, {n(P1['median_pred_A'])} on COMPAS\u2019s 1\u201310 scale, so it flags {pct(100 * P1['flag_rate_A'], 2)} of defendants by construction. "
        f"Model B\u2019s 0.5 is an absolute probability; because only {fr(S['base_rate_test'])} of test defendants reoffended, it flags {pct(100 * P1['flag_rate_B'], 2)}. "
        f"Second, the targets differ. Model A imitates COMPAS\u2019s proprietary score and adds {n(coefA['race_African_American']['coef'])} decile points for African-American defendants (p {pv(coefA['race_African_American']['p'])}), "
        f"whereas Model B, trained on real outcomes, finds a log-odds gap of only {n(coefB['race_African_American']['coef'])} (p {pv(coefB['race_African_American']['p'])}). "
        f"Third, a linear model is unbounded: Model A predicts from {n(A['pred_min'], 2)} to {n(A['pred_max'], 2)}. "
        f"The median rule also changes the error profile: Model A\u2019s flag has higher recall ({fr(va['recall'])} versus {fr(vb['recall'])}) but lower precision ({fr(va['precision'])} versus {fr(vb['precision'])}) and a false-positive rate of {fr(va['FPR'])} against {fr(vb['FPR'])}. "
        f"Rankings still agree strongly (Spearman {n(P1['spearman_A_B'])}), so mismatches cluster near the boundary: their mean Model B probability is {n(P1['mismatch_mean_pB'])} (range {n(P1['mismatch_min_pB'])}\u2013{n(P1['mismatch_max_pB'])}). "
        f"Cutting Model B at {n(P1['pB_matched_threshold'])}, which flags the same share as Model A, reduces mismatches to {P1['mismatch_matched_n']} ({pct(P1['mismatch_matched_pct'], 2)}); "
        f"so {mm - P1['mismatch_matched_n']} disagreements ({pct(100 * (mm - P1['mismatch_matched_n']) / mm, 0)}) are pure threshold semantics and the rest reflects genuinely different models. "
        f"Agreement is not accuracy: {P1['agree_but_both_wrong']} defendants ({pct(100 * P1['agree_but_both_wrong'] / P1['n_test'], 1)}) receive identical but wrong flags. Against actual outcomes Model A\u2019s flag is {fr(P1['vs_actual']['Model A flag (>= median)']['accuracy'])} accurate and Model B\u2019s "
        f"{fr(P1['vs_actual']['Model B flag (p >= 0.5)']['accuracy'])}; on the disagreements Model B was right {P1['mismatch_B_right']} times, Model A {P1['mismatch_A_right']}."
    )
    # ---- Q2
    under = 100 * (pri["B_coef"] - pri["full_coef"]) / pri["B_coef"]
    Q[2] = (
        f"I split by sex (A = male, n = {sx['n_A']:,}; B = female, n = {sx['n_B']:,}). No strict Simpson-style reversal exists: of {R['p21_scan']['n_tests']} scanned subgroup comparisons, {R['p21_scan']['n_sign_flips']} changed sign, but in none were both coefficients significantly non-zero. "
        f"The robust finding is a stronger prior-record effect for women. Per standard deviation of priors_count the logit coefficient is {n(pri['full_coef'])} in the full data (odds ratio {pri['OR_full']:.2f}), "
        f"{n(pri['A_coef'])} {ci(pri['A_lo'], pri['A_hi'])} for men (OR {pri['OR_A']:.2f}) and {n(pri['B_coef'])} {ci(pri['B_lo'], pri['B_hi'])} for women (OR {pri['OR_B']:.2f}). The intervals do not overlap; the difference test gives "
        f"z = {n(abs(pri['diff_z']), 2)}, p = {pri['diff_p']:.4f}, and p = {pri['diff_p_bonferroni']:.3f} after Bonferroni correction over {R['p21_scan']['n_tests']} tests. "
        f"Why? Much of the gap is functional form rather than a different mechanism. Women have fewer priors (mean {n(sx['priors_by_sex']['Female']['mean'], 2)}, median {sx['priors_by_sex']['Female']['median']:.0f}, versus {n(sx['priors_by_sex']['Male']['mean'], 2)} and {sx['priors_by_sex']['Male']['median']:.0f}), "
        f"so they sit on the steep part of a concave risk curve; with log(1 + priors) the coefficients converge to {n(lg['male_beta'])} versus {n(lg['female_beta'])} (p = {lg['diff_p']:.3f}). "
        f"The pooled model\u2019s test calibration shows the consequence: women with at least five priors (n = {cal['Female|priors>=5']['n']}) had predicted risk {n(cal['Female|priors>=5']['mean_pred'])} against {n(cal['Female|priors>=5']['observed'])} observed, "
        f"and women with at most one prior (n = {cal['Female|priors<=1']['n']}) {n(cal['Female|priors<=1']['mean_pred'])} against {n(cal['Female|priors<=1']['observed'])}. "
        f"If the subgroup were ignored, the pooled coefficient understates priors for women by about {under:.0f}%, so a rule such as \u201cflag five or more priors\u201d would under-prioritise high-prior women and over-flag low-prior women. "
        f"However, sex-specific models did not improve out-of-sample discrimination for women (AUC {n(perf['Female']['auc_group_specific'])} versus {n(perf['Female']['auc_pooled'])} pooled), so the policy change is to audit calibration by sex, not to build separate scorecards. "
        f"Splitting by race gives priors coefficients of {n(rct['priors_count']['A_coef'])} (African-American) versus {n(rct['priors_count']['B_coef'])} (Caucasian), nominal p = {rct['priors_count']['diff_p']:.3f} but Bonferroni p = {rct['priors_count']['diff_p_bonferroni']:.2f}, so I do not claim it."
    )
    # ---- Q3
    rr = o22["race_African_American"]
    Q[3] = (
        f"I dropped <b>priors_count</b>, the strongest predictor (\u03b2 = {n(p22['priors_coef_full'])}) and one correlated with race_African_American (r = {n(p22['corr_with_priors']['race_African_American'], 2)}), "
        f"juv_misd_count ({n(p22['corr_with_priors']['juv_misd_count'], 2)}) and age ({n(p22['corr_with_priors']['age'], 2)}). [[tbl:p22q3]] shows the before/after coefficients, with %\u0394 = (\u03b2<sub>reduced</sub> \u2212 \u03b2<sub>full</sub>)/|\u03b2<sub>full</sub>| \u00d7 100. "
        f"By percentage, juv_misd_count moved most ({n(o22['juv_misd_count']['beta_full'])} \u2192 {n(o22['juv_misd_count']['beta_reduced'])}, {o22['juv_misd_count']['pct_change']:+.0f}%, a sign flip), but its baseline is almost zero, so that figure is a denominator artefact. "
        f"Measured in absolute terms and in standard errors, age shifted most ({n(o22['age']['beta_full'])} \u2192 {n(o22['age']['beta_reduced'])}, {o22['age']['pct_change']:+.1f}%, {n(o22['age']['change_in_SE_units'], 1)} SE). "
        f"The most consequential shift is race_African_American, from {n(rr['beta_full'])} (p {pv(rr['p_full'])}) to {n(rr['beta_reduced'])} (p {pv(rr['p_reduced'])}), {rr['pct_change']:+.1f}%: the odds ratio rises from {2.718281828 ** rr['beta_full']:.2f} to {2.718281828 ** rr['beta_reduced']:.2f}. "
        f"Other shifts: juv_fel_count {n(o22['juv_fel_count']['beta_full'])} \u2192 {n(o22['juv_fel_count']['beta_reduced'])} ({o22['juv_fel_count']['pct_change']:+.0f}%), felony {o22['felony']['pct_change']:+.1f}% and male {o22['male']['pct_change']:+.1f}%. "
        f"The direction follows the omitted-variable logic: priors correlate positively with age and raise risk, so removing them weakens age\u2019s protective coefficient. "
        f"Four coefficients became significant and none lost significance, and test AUC fell from {n(p22['perf_full']['auc'])} to {n(p22['perf_reduced']['auc'])}. "
        f"The lesson is that a coefficient is conditional on the other variables in the model: the reduced model would tell a policy-maker that race predicts recidivism after adjustment, while the full model finds no significant race effect. "
        f"Both models use the same {S['n_train']:,} training rows, so the shifts are caused by the omission alone. "
        f"Two caveats apply: logistic coefficients are non-collapsible, so some change occurs even without confounding, and priors may be a mediator of policing exposure, so neither model is causal. "
        f"A model whose coefficients have never been stress-tested this way is a snapshot of correlations that can rearrange when one variable is absent, so I would not read any single coefficient, least of all a race coefficient, at face value."
    )
    # ---- Q4
    Q[4] = (
        f"As a practitioner I would run a <b>covariate-augmentation and sensitivity audit</b> rather than trust the current feature set. "
        f"Step 1: convene a domain-expert audit, with public defenders, probation officers and a county data steward, to list factors that plausibly drive both exposure to the justice system and reoffending but are absent from the file: employment, housing stability, "
        f"substance-use treatment, supervision conditions and neighbourhood policing intensity. Step 2: obtain proxies for a pilot sample, for example census-tract arrest rates and unemployment joined by address, collected under data-protection approval. "
        f"Step 3: refit the model with each candidate and measure coefficient shifts in standard-error units. My own experiment supplies the yardstick: omitting priors_count moved age by {n(o22['age']['change_in_SE_units'], 1)} SE and race_African_American by {n(o22['race_African_American']['change_in_SE_units'], 1)} SE. "
        f"I would pre-register a trigger, for example any key or protected-attribute coefficient moving by more than 1 SE, which would send the model back for redesign. "
        f"Step 4: for confounders that cannot be measured, quantify robustness with an omitted-variable sensitivity analysis (Cinelli and Hazlett, 2020) or an E-value (VanderWeele and Ding, 2017), asking how strong an unmeasured variable would have to be, relative to priors_count, "
        f"to erase the effect of interest. Step 5: validate externally on another county or year and compare calibration within groups. "
        f"Limitations are real: experts only name confounders they can imagine, proxies are noisy, and the audit costs time and data access. It is also unable to prove absence of confounding; it can only bound it. "
        f"I would still prefer it to a single hold-out accuracy figure, which in this dataset ({fr(B['accuracy'])}) said nothing about the {p22['n_sig_gain']} coefficients that changed significance when one variable disappeared."
    )
    # ---- Q5
    zp = p23["zprofile"]
    contrib_fp = coefB["priors_count"]["coef"] * zp["FP rows"]["priors_count"]
    contrib_fn_age = coefB["age"]["coef"] * zp["FN rows"]["age"]
    Q[5] = (
        f"The {p23['n_subset']}-row subset is {p23['subset_n_fp']} false positives and {p23['subset_n_fn']} false negatives, and the shared trait is <b>age together with prior record at the extremes</b>. "
        f"None of the 50 is under 25 (test set: {pct(te_p['pct_age_lt25'])}); {pct(sub_p['pct_age_ge45'], 0)} are 45 or older (test set {pct(te_p['pct_age_ge45'])}). Within the subset, <b>{pct(fp_p['pct_priors_ge5'], 0)} of the false positives have five or more priors</b> "
        f"(median {fp_p['priors_median']:.0f} versus {te_p['priors_median']:.0f} in the test set), while <b>{pct(fn_p['pct_priors_le1'], 0)} of the false negatives have at most one prior</b> and {pct(fn_p['pct_age_ge45'], 0)} are 45 or older; none has a juvenile record. "
        f"The mechanism is linear extrapolation. In Model B each standard deviation of priors adds {n(coefB['priors_count']['coef'])} log-odds and each of age subtracts {n(abs(coefB['age']['coef']))}. The false positives average "
        f"+{n(zp['FP rows']['priors_count'], 2)} SD of priors, a contribution of +{n(contrib_fp, 2)} log-odds, so they sit at mean predicted probability {n(fp_p['mean_pred_prob'], 2)}; the false negatives average +{n(zp['FN rows']['age'], 2)} SD of age "
        f"({n(contrib_fn_age, 2)} log-odds) and {n(zp['FN rows']['priors_count'], 2)} SD of priors, giving {n(fn_p['mean_pred_prob'], 2)}. Both sit in tails where the straight-line assumption is weakest: the extended model finds priors\u00b2 = {n(ext['coef']['priors_sq']['coef'])} "
        f"(risk saturates) and age\u00b2 = +{n(ext['coef']['age_sq']['coef'])} (old age protects less than linear). Yet non-linearity is not the main story. Calibration shows {p23['n_qualifying']} confident errors against {ex_tot:.1f} expected from the model\u2019s own probabilities, "
        f"and the extended model raises test AUC only from {n(ext['auc_base'])} to {n(ext['auc_ext'])} while making {ext['n_conf_errors_ext']} confident errors. Most of these failures are outcome noise that no feature in the file can predict. "
        f"Fisher exact tests against the remaining test rows confirm both patterns (priors \u2265 5: {pct(trait[('priors >= 5', 'FP only')]['pct_in_group'], 0)} versus {pct(trait[('priors >= 5', 'FP only')]['pct_in_rest_of_test'])}; age \u2265 45: "
        f"{pct(trait[('age >= 45', 'FN only')]['pct_in_group'])} versus {pct(trait[('age >= 45', 'FN only')]['pct_in_rest_of_test'])}; both p &lt; 0.001). The errors also have a group pattern: {pct(fp_p['pct_African_American'], 0)} of the false positives are African-American, whereas {pct(fn_p['pct_Caucasian'], 0)} of the false negatives are Caucasian."
    )
    # ---- Q6
    s25, s35, s45, s55 = sw[0.25], sw[0.35], sw[0.45], sw[0.55]
    Q[6] = (
        f"Consider two people who would <i>not</i> reoffend: Ms A (Caucasian) and Mr B (African-American). Using the sweep in [[fig:tradeoff]], the chance that each is wrongly labelled high risk is {fr(s55['FPR_A'])} versus {fr(s55['FPR_B'])} at t = 0.55, "
        f"{fr(s45['FPR_A'])} versus {fr(s45['FPR_B'])} at t = 0.45, {fr(s35['FPR_A'])} versus {fr(s35['FPR_B'])} at t = 0.35 and {fr(s25['FPR_A'])} versus {fr(s25['FPR_B'])} at t = 0.25. "
        f"The sweep\u2019s EO (the FPR ratio) improves from {n(s55['EO'], 2)} to {n(s25['EO'], 2)}, but only because Ms A\u2019s false-positive chance rises almost tenfold; parity is bought by flagging more innocent people, {fp_tot[0.25]} across both groups at t = 0.25 against {fp_tot[0.55]} at t = 0.55. "
        f"Prioritising precision and accuracy (t = 0.55, PP {n(s55['PP'], 2)}) has the opposite cost: Mr B\u2019s chance of an unwarranted flag is {s55['FPR_B'] / s55['FPR_A']:.1f} times Ms A\u2019s, "
        f"while {fr(s55['FN_A'] / M['n_pos_A'])} of Caucasian reoffenders go unflagged against {fr(s55['FN_B'] / M['n_pos_B'])} of African-American reoffenders. "
        f"At t = {ts:.2f} that means {M['fp_B_at_star']} of {M['n_neg_B']} African-American non-reoffenders and {M['fp_A_at_star']} of {M['n_neg_A']} Caucasian non-reoffenders were wrongly flagged. "
        f"Identical records are treated almost identically: a hypothetical 30-year-old man with three priors and a felony charge receives P = {n(ill30_aa, 2)} if African-American and {n(ill30_c, 2)} if Caucasian, so both are flagged at t = 0.45 and neither at 0.55. "
        f"The group gap therefore arises from different record distributions (mean priors {n(M['priors_mean_B'], 2)} for Group B versus {n(M['priors_mean_A'], 2)} for Group A; mean age {n(M['age_mean_B'], 1)} versus {n(M['age_mean_A'], 1)}), "
        f"which may themselves reflect unequal policing. No threshold removes the harm; each moves it between people. A judge reading a flag should know that, at t = {ts:.2f}, a non-reoffending African-American defendant faces a {fr(star['FPR_B'])} chance of that label."
    )
    # ---- Q7
    Q[7] = (
        f"I choose <b>t = {ts:.2f}</b>. The defence rests on six numbers. Accuracy is {fr(star['acc_all'])} ({fr(star['acc_A_B'])} within the two groups analysed), only {100 * (best['acc_all'] - star['acc_all']):.2f} points below the best ({fr(best['acc_all'])} at t = {tb:.2f}) and {100 * (star['acc_all'] - S['majority_baseline_acc_test']):.1f} points above the {fr(S['majority_baseline_acc_test'])} majority baseline. "
        f"Relative to t = {tb:.2f}, DP improves from {n(best['DP'])} to {n(star['DP'])} (+{p1_['DP_gain_pct']:.1f}%) and EO from {n(best['EO'])} to {n(star['EO'])} (+{p1_['EO_gain_pct']:.1f}%) for that small accuracy cost. "
        f"PP is {n(star['PP'])} with a bootstrap interval {ci(sci['PP_lo'], sci['PP_hi'])} that contains 1, so a \u201chigh-risk\u201d label is about equally likely to be correct for both groups ({fr(star['PPV_A'])} versus {fr(star['PPV_B'])}). "
        f"The choice follows a stated rule: the highest worst-case parity score among thresholds within two accuracy points of the best, and it is stable for tolerances of one to five points. "
        f"Lower thresholds look fairer on ratios but flag {fr(sw[0.35]['flag_rate_B'])} of Group B at t = 0.35 and {fr(sw[0.25]['flag_rate_B'])} at 0.25. "
        f"<b>Strongest counterargument:</b> EO = {n(star['EO'])} means non-reoffending African-American defendants are flagged {star['FPR_B'] / star['FPR_A']:.2f} times as often ({fr(star['FPR_B'])} versus {fr(star['FPR_A'])}; {M['fp_B_at_star']} versus {M['fp_A_at_star']} people), "
        f"and the intervals for DP {ci(sci['DP_lo'], sci['DP_hi'])} and EO {ci(sci['EO_lo'], sci['EO_hi'])} exclude 1. A critic could say a threshold with that disparity is not fair in any practical sense, and could point to t = 0.35 (EO {n(sw[0.35]['EO'])}). "
        f"Fairness also cuts the other way here: at this threshold the model catches {fr(star['TPR_B'])} of African-American reoffenders but only {fr(star['TPR_A'])} of Caucasian reoffenders. I accept the objection\u2019s validity. t = 0.35 does improve EO, but at an accuracy cost of {abs(p2_['acc_change_pts']):.1f} points and with {fp_tot[0.35]} wrongly flagged people instead of {fp_tot[0.45]}. My choice is a values judgment about that exchange rate, not a fact the data can settle."
    )
    # ---- Q8
    dp_up = 100 * (s25["DP"] / star["DP"] - 1)
    eo_up = 100 * (s25["EO"] / star["EO"] - 1)
    Q[8] = (
        f"I would recommend prioritising <b>equalized odds</b>, specifically parity in false-positive rates, over demographic parity. "
        f"The groups\u2019 base rates differ ({fr(M['base_rate_B'])} versus {fr(M['base_rate_A'])} in the test set), so demographic parity would require equal flag rates regardless of what people did. At t = {ts:.2f} the flag rates are {fr(star['flag_rate_B'])} (B) and {fr(star['flag_rate_A'])} (A), "
        f"and closing that gap by lowering the threshold to 0.25 (DP {n(star['DP'])} \u2192 {n(s25['DP'])}, +{dp_up:.0f}%) means flagging {fr(s25['flag_rate_B'])} of Group B and {fr(s25['flag_rate_A'])} of Group A: mass flagging, with overall accuracy falling to {fr(s25['acc_all'])}. "
        f"Equalized odds conditions on the actual outcome, so it targets the concrete harm in this setting: non-reoffenders being wrongly detained or denied release, a direct loss of liberty whose burden can be audited from outcomes alone. The false-positive ratio is {n(star['FPR_ratio'])} at t = {ts:.2f} and {n(s25['FPR_ratio'])} at t = 0.25. "
        f"In this model the two metrics rise together (r = {n(M['corr_DP_EO'])}; EO +{eo_up:.0f}% over the same move), so choosing EO does not sacrifice DP; what is sacrificed is predictive parity and accuracy. PP slips from {n(star['PP'])} to {n(s25['PP'])} ({abs(100 * (s25['PP'] / star['PP'] - 1)):.1f}% lower) "
        f"and accuracy by {100 * (star['acc_all'] - s25['acc_all']):.1f} points. I find that acceptable because PP stays within {100 * (1 - s25['PP']):.0f}% of parity, whereas a {n(star['FPR_ratio'], 2)} false-positive ratio is not. "
        f"Two cautions: EO, defined as a minimum of ratios, can hide a large gap in the weaker ratio; and a race-blind model still gives DP {n(R['robust']['Race-blind (no race indicators)']['DP@0.45'])} and EO {n(R['robust']['Race-blind (no race indicators)']['EO@0.45'])} at t = 0.45, "
        f"so the disparity runs through proxies such as priors and age, and policy must also scrutinise those inputs. I would write this into policy as a published ceiling on the false-positive ratio, audited annually."
    )
    T["Q"] = Q

    # ================================================================================== PART 4
    T["refl_direct"] = (
        f"<b>Direct answer.</b> I would recommend <b>conditional deployment</b>: advisory decision support only, at t = {ts:.2f}, with mandatory group-level monitoring, and not as an automated trigger for detention or sentencing. "
        f"The headline reason is that {fr(star['acc_all'])} overall accuracy conceals a false-positive rate of {fr(star['FPR_B'])} for African-American versus {fr(star['FPR_A'])} for Caucasian defendants."
    )
    gA, gB = race_perf["Caucasian"]["auc_pooled"], race_perf["African-American"]["auc_pooled"]
    T["refl_fail"] = (
        f"<b>Why a single number fails.</b> First, <i>overall accuracy hides subgroup gaps</i>. The headline {fr(B['accuracy'])} at 0.5 says nothing about who bears the errors: at t = {ts:.2f} the false-positive ratio is {n(star['FPR_ratio'], 2)} and the true-positive ratio {n(star['TPR_ratio'], 2)}, "
        f"and even discrimination differs by group (test AUC {n(gB)} for African-American versus {n(gA)} for Caucasian defendants); the pooled model also under-predicts Caucasian risk ({n(cal_r['Caucasian|all']['mean_pred'], 3)} predicted versus {n(cal_r['Caucasian|all']['observed'], 3)} observed). "
        f"Second, <i>the threshold is a values question, not a data-driven fact</i>. Accuracy peaks at t = {tb:.2f} ({fr(best['acc_all'])}); the ratio metrics are \u201cperfect\u201d only at t = 0.05, where everyone is flagged and accuracy ({fr(sw[0.05]['acc_all'])}) falls below the {fr(S['majority_baseline_acc_test'])} baseline. "
        f"My own rule picks {ts:.2f} for any accuracy tolerance from one to five points and {tol['0.07']:.2f} from seven to ten, so the answer depends on a human choice about how much accuracy to trade. Even the conventional 0.5 is a choice: it misses {fr(1 - B['recall'])} of reoffenders. "
        f"Third, <i>fairness metrics conflict</i>. Base rates differ ({fr(M['base_rate_B'])} versus {fr(M['base_rate_A'])}), so predictive parity ({n(sw[0.55]['PP'], 2)} at t = 0.55) and equalized odds ({n(sw[0.55]['EO'], 2)} at the same threshold) cannot both hold; "
        f"improving EO to {n(sw[0.25]['EO'], 2)} at t = 0.25 flags {fp_tot[0.25]} non-reoffenders instead of {fp_tot[0.55]}. "
        f"Fourth, <i>the model fails confidently on predictable subsets</i>: {p23['n_qualifying']} test cases ({pct(p23['pct_of_test'], 1)}) are confident errors, concentrated among defendants with five or more priors (false positives) and older defendants with almost none (false negatives), "
        f"and a single accuracy number gives no hint of where. Fifth, <i>hidden confounds remain unknowable</i>: omitting one variable moved the race coefficient from {n(rr['beta_full'])} to {n(rr['beta_reduced'])} and made {p22['n_sig_gain']} coefficients significant, "
        f"so any variable I have not measured could do the same. Finally, <i>every ratio carries sampling error</i>: with {M['n_A']} Caucasian test defendants, DP at t = {ts:.2f} has a 95% bootstrap interval of {ci(sci['DP_lo'], sci['DP_hi'])} and PP {ci(sci['PP_lo'], sci['PP_hi'])}, which cannot distinguish parity from a gap of roughly 18%. "
        f"A policy-maker given one number would inherit all of these problems without being able to see them."
    )
    T["refl_ground"] = (
        f"<b>Grounding in evidence.</b> Part 1: {mm} mismatched flags ({pct(P1['mismatch_pct'], 2)}), median {n(P1['median_pred_A'])} versus the 0.5 threshold. Part 2.1: priors effect {n(pri['B_coef'])} (women) versus {n(pri['A_coef'])} (men), p = {pri['diff_p']:.4f}. "
        f"Part 2.2: race coefficient {n(rr['beta_full'])} \u2192 {n(rr['beta_reduced'])}. Part 2.3: {p23['n_qualifying']} confident errors against {ex_tot:.1f} expected ({p23['subset_n_fp']} false positives and {p23['subset_n_fn']} false negatives in the subset). Part 2.4 at t = {ts:.2f}: DP {n(star['DP'])}, EO {n(star['EO'])}, PP {n(star['PP'])}, accuracy {fr(star['acc_all'])}."
    )
    T["refl_close"] = (
        "<b>Closing.</b> The meta-lesson is that responsible deployment means publishing the trade-offs, not hiding them inside one metric. Decision-makers should see the threshold, the error rates by group, the confident failures and the unresolved confounds, "
        "and be able to challenge each. A model card carrying these tables, rather than a score alone, is what I would hand to the policy-maker."
    )

    # ================================================================================== LIMITATIONS & CONCLUSION
    T["limits"] = (
        f"<b>Limitations.</b> (1) The target is two-year re-arrest recorded in one Florida county, which measures policing as well as behaviour; label bias cannot be tested with this file. "
        f"(2) The analysis uses one random split (seed 450); the fairness estimates for Group A rest on {M['n_A']} test defendants, so bootstrap intervals are wide (for example DP at t = {ts:.2f}: {ci(sci['DP_lo'], sci['DP_hi'])}) and values at t \u2265 0.85 rest on very few flagged people. "
        f"(3) Section 2.1 was a search, so I applied a Bonferroni correction across {R['p21_scan']['n_tests']} comparisons and selected the reported split after the scan; the selection is declared, not hidden. "
        f"(4) Race was included as a predictor to expose its effect; removing it did not remove the disparity (Appendix B). (5) Logistic coefficients are non-collapsible and priors may be a mediator, so no causal claim is made. "
        f"(6) The recommended threshold depends on a stated 2-point accuracy tolerance. (7) The EO definition (a minimum of ratios) can mask the weaker disparity; [[tbl:fair_main]] reports both ratios. "
        f"<b>Conclusion.</b> A simple, transparent logistic model predicts two-year recidivism with accuracy {fr(B['accuracy'])} and AUC {n(B['auc'])}, yet every stress test exposed something the headline metrics hide: disagreement with the linear score "
        f"({pct(P1['mismatch_pct'], 2)}), coefficient fragility, structured confident errors and unequal error rates across groups. The defensible conclusion is conditional, advisory use with explicit, published trade-offs."
    )

    T["refs"] = [
        "Angwin, J., Larson, J., Mattu, S., &amp; Kirchner, L. (2016). Machine bias. <i>ProPublica</i>, 23 May 2016.",
        "Barocas, S., Hardt, M., &amp; Narayanan, A. (2023). <i>Fairness and Machine Learning: Limitations and Opportunities</i>. MIT Press.",
        "Bickel, P. J., Hammel, E. A., &amp; O\u2019Connell, J. W. (1975). Sex bias in graduate admissions: Data from Berkeley. <i>Science</i>, 187(4175), 398\u2013404.",
        "Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. <i>Big Data</i>, 5(2), 153\u2013163.",
        "Cinelli, C., &amp; Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. <i>Journal of the Royal Statistical Society: Series B</i>, 82(1), 39\u201367.",
        "Dieterich, W., Mendoza, C., &amp; Brennan, T. (2016). <i>COMPAS risk scales: Demonstrating accuracy equity and predictive parity</i>. Northpointe Inc.",
        "Dressel, J., &amp; Farid, H. (2018). The accuracy, fairness, and limits of predicting recidivism. <i>Science Advances</i>, 4(1), eaao5580.",
        "Hardt, M., Price, E., &amp; Srebro, N. (2016). Equality of opportunity in supervised learning. <i>Advances in Neural Information Processing Systems</i>, 29.",
        "Kleinberg, J., Mullainathan, S., &amp; Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. <i>Proceedings of ITCS 2017</i>, LIPIcs 67, 43:1\u201343:23.",
        "Larson, J., Mattu, S., Kirchner, L., &amp; Angwin, J. (2016). How we analyzed the COMPAS recidivism algorithm. <i>ProPublica</i>, 23 May 2016. Data: github.com/propublica/compas-analysis.",
        "Mood, C. (2010). Logistic regression: Why we cannot do what we think we can do, and what we can do about it. <i>European Sociological Review</i>, 26(1), 67\u201382.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825\u20132830.",
        "Seabold, S., &amp; Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. <i>Proceedings of the 9th Python in Science Conference</i>.",
        "Simpson, E. H. (1951). The interpretation of interaction in contingency tables. <i>Journal of the Royal Statistical Society: Series B</i>, 13(2), 238\u2013241.",
        "VanderWeele, T. J., &amp; Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. <i>Annals of Internal Medicine</i>, 167(4), 268\u2013274.",
    ]
    return T


def plain(s):
    return re.sub(r"<[^>]+>", "", s).replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")


if __name__ == "__main__":
    T = build(load())
    print("--- Part 3 word counts (target 250-300) ---")
    for k, v in T["Q"].items():
        print(f"Q{k}: {words(plain(v))}")
    refl = [T["refl_direct"], T["refl_fail"], T["refl_ground"], T["refl_close"]]
    print("Reflection sections:", [words(plain(x)) for x in refl], "total", sum(words(plain(x)) for x in refl))
    for k in ["abstract", "intro", "dataset", "p21", "p22", "p23", "p24_tradeoff", "limits"]:
        print(k, words(plain(T[k])))
