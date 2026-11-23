"""CODE-02: US type 2 diabetes benchmark case shares (DATA-02) and their survey errors.

Run:  python CODE-02-benchmarks.py
Inputs:
  DATA-03-nhanes-person-level.parquet   NHANES 2017 to March 2020 prepandemic, adults 20+
  DATA-04-acs-adult20-frame.json        ACS 2019 to 2023 five-year population counts
  nhanes/P_DIQ.xpt, nhanes/P_GLU.xpt    downloaded from the CDC website on first run
Writes RESULTS-10-benchmarks.csv, which has the same rows and columns as DATA-02 and is
compared with it at the end of the run; RESULTS-12-allocated-share-rse.csv; and
RESULTS-11-widened-intervals-both-error-sets.csv (Table S2 under both error sets).
Also needs DATA-01 and CODE-01-reproduce.py for the widened intervals.

Primary race frame: NHANES weighted diabetes case counts for five survey groups (Hispanic,
and non-Hispanic White, Black, Asian, and other or multiple race) are mapped to the six OMB
race categories. Non-Hispanic White, Black, and Asian cases go to the matching race.
Non-Hispanic other or multiple race cases are split across American Indian or Alaska Native,
Native Hawaiian or Other Pacific Islander, and other or multiple race by their ACS
non-Hispanic adult populations. Hispanic cases are split across all six categories by the ACS
race composition of Hispanic adults. Alternative frames assign all Hispanic cases to White
(hispanic_as_white) or drop them (nonhispanic_only).

Survey errors are Taylor-linearized standard errors of each case share under the NHANES
stratified cluster design (SDMVSTRA, SDMVPSU), reported as relative standard errors in percent
for the survey-group and ethnicity shares.

This is the type 2 diabetes part of the original benchmark script, which also built
benchmarks for other conditions in companion work. The diabetes code is unchanged.
"""
import json, os, sys, urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def ratio_and_se(d, w, num, den):
    """R = sum(w*num)/sum(w*den); linearized SE using strata/PSU. num, den are 0/1 or float arrays."""
    w = np.asarray(w, float); num = np.asarray(num, float); den = np.asarray(den, float)
    ok = np.isfinite(w) & (w > 0) & np.isfinite(num) & np.isfinite(den)
    w, num, den = w[ok], num[ok], den[ok]
    st = np.asarray(d.SDMVSTRA, float)[ok]; psu = np.asarray(d.SDMVPSU, float)[ok]
    Y = float((w * num).sum()); X = float((w * den).sum())
    if X <= 0:
        return np.nan, np.nan, Y, X
    R = Y / X
    z = (num - R * den) / X
    var = 0.0
    for h in np.unique(st):
        m = st == h
        tot = np.array([ (w[m & (psu == p)] * z[m & (psu == p)]).sum() for p in np.unique(psu[m]) ])
        nh = len(tot)
        if nh > 1:
            var += nh / (nh - 1.0) * ((tot - tot.mean()) ** 2).sum()
    return R, float(np.sqrt(var)), Y, X


def wtotal(w, ind):
    w = np.asarray(w, float); ind = np.asarray(ind, float)
    ok = np.isfinite(w) & (w > 0) & np.isfinite(ind)
    return float((w[ok] * ind[ok]).sum())


def load_nhanes(parquet_path, diq_path, glu_path):
    d = pd.read_parquet(parquet_path)
    diq = pd.read_sas(diq_path)[["SEQN", "DID040"]]
    glu = pd.read_sas(glu_path)[["SEQN", "WTSAFPRP", "LBXGLU"]]
    d = d.merge(diq, on="SEQN", how="left").merge(glu, on="SEQN", how="left")
    # DID040: age when first told had diabetes. 666 = less than 1 year, 999/. = refused/DK
    d["dx_age"] = d.DID040.where(d.DID040 < 100)
    return d


def define_conditions(d):
    """Attach one 0/1 case indicator per diabetes definition."""
    selfrep_dm = (d.DIQ010 == 1)
    med_dm = (d.DIQ050 == 1) | (d.DIQ070 == 1)
    a1c_dm = d.LBXGH >= 6.5
    fpg_dm = d.LBXGLU >= 126

    # primary composite needs A1c measured or a self-report/medication answer
    dm_obs = d.LBXGH.notna() | d.DIQ010.isin([1, 2])
    d["dm_primary"] = np.where(dm_obs, (a1c_dm | selfrep_dm | med_dm).astype(float), np.nan)
    d["dm_a1c_only"] = np.where(d.LBXGH.notna(), a1c_dm.astype(float), np.nan)
    d["dm_selfreport_only"] = np.where(d.DIQ010.isin([1, 2]), selfrep_dm.astype(float), np.nan)
    fpg_obs = d.LBXGLU.notna()
    d["dm_fpg_composite"] = np.where(fpg_obs, (a1c_dm | selfrep_dm | med_dm | fpg_dm).astype(float), np.nan)
    # likely type 1: diagnosed before age 30, on insulin, not on oral glucose-lowering agents
    likely_t1 = (d.dx_age < 30) & (d.DIQ050 == 1) & (d.DIQ070 != 1)
    d["dm_type1_excluded"] = np.where(dm_obs, ((a1c_dm | selfrep_dm | med_dm) & ~likely_t1).astype(float), np.nan)
    return d


NHANES_GROUP = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
                4: "NH Black", 6: "NH Asian", 7: "NH Other or multiple"}
ETH = ["Hispanic or Latino", "Not Hispanic or Latino"]


def nhanes_case_counts(d, flag, weight):
    """Weighted cases by NHANES survey group and by Hispanic ethnicity, with linearized SEs
    of the case shares."""
    w = d[weight]
    obs = d[flag].notna() & w.notna() & (w > 0)
    dd = d[obs].copy(); ww = dd[weight].values; y = dd[flag].values
    out = {}
    grp = dd.RIDRETH3.map(NHANES_GROUP)
    out["cases_by_group"] = pd.Series({g: wtotal(ww, y * (grp == g).values) for g in NHANES_GROUP.values()})
    out["se_share_by_group"] = pd.Series(
        {g: ratio_and_se(dd, ww, y * (grp == g).values, y)[1] for g in NHANES_GROUP.values()})
    eth = np.where(dd.RIDRETH3.isin([1, 2]).values, ETH[0], ETH[1])
    out["cases_by_eth"] = pd.Series({e: wtotal(ww, y * (eth == e)) for e in ETH})
    out["se_share_by_eth"] = pd.Series({e: ratio_and_se(dd, ww, y * (eth == e), y)[1] for e in ETH})
    return out


RACE6 = ["White", "Black or African American", "Asian",
         "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander",
         "Other or multiple race"]
def build_population_frame(acs):
    """Adult 20+ population in a race x ethnicity frame.
    Race categories are OMB race-alone/multiple (every person has a race), ethnicity is separate.
    White-Hispanic is exact (B01001A minus B01001H); the remaining Hispanic races are split by
    the B03002 all-age composition, the only ACS table crossing Hispanic origin with race."""
    T = lambda t: acs[t]["male"] + acs[t]["female"]
    race_tab = {"White": "B01001A", "Black or African American": "B01001B",
                "Asian": "B01001D", "American Indian or Alaska Native": "B01001C",
                "Native Hawaiian or Other Pacific Islander": "B01001E"}
    pop_race = {k: T(v) for k, v in race_tab.items()}
    pop_race["Other or multiple race"] = T("B01001F") + T("B01001G")
    hisp_total = T("B01001I")
    hisp_white = T("B01001A") - T("B01001H")
    B = acs["B03002"]
    resid_keys = {"Black or African American": "B03002_014E",
                  "American Indian or Alaska Native": "B03002_015E",
                  "Asian": "B03002_016E",
                  "Native Hawaiian or Other Pacific Islander": "B03002_017E"}
    resid_other = B["B03002_018E"] + B["B03002_019E"]
    resid_tot = sum(B[v] for v in resid_keys.values()) + resid_other
    hisp_resid = hisp_total - hisp_white
    hisp_race = {"White": hisp_white}
    for k, v in resid_keys.items():
        hisp_race[k] = hisp_resid * B[v] / resid_tot
    hisp_race["Other or multiple race"] = hisp_resid * resid_other / resid_tot
    nonhisp_race = {k: pop_race[k] - hisp_race[k] for k in RACE6}
    return {"pop_race": pop_race, "hisp_race": hisp_race, "nonhisp_race": nonhisp_race,
            "hisp_total": hisp_total, "nonhisp_total": sum(nonhisp_race.values()),
            "male": sum(acs[f"B01001{g}"]["male"] for g in "ABCDEFG"),
            "female": sum(acs[f"B01001{g}"]["female"] for g in "ABCDEFG")}


def allocate_to_omb_race(cases_by_group, frame, mode="omb_allocated"):
    """cases_by_group indexed by NHANES_GROUP values. Returns dict race -> cases.
    mode:
      omb_allocated          Hispanic cases split across races by ACS Hispanic race composition
      nonhispanic_only       Hispanic cases dropped (race shares among non-Hispanic adults)
      hispanic_as_white      all Hispanic cases assigned to White
    Non-Hispanic 'other or multiple' NHANES cases are split across AIAN, NHPI and other/multiple
    by the ACS non-Hispanic composition of those three categories.
    """
    hisp_cases = cases_by_group["Mexican American"] + cases_by_group["Other Hispanic"]
    out = {r: 0.0 for r in RACE6}
    out["White"] += cases_by_group["NH White"]
    out["Black or African American"] += cases_by_group["NH Black"]
    out["Asian"] += cases_by_group["NH Asian"]
    resid = ["American Indian or Alaska Native",
             "Native Hawaiian or Other Pacific Islander", "Other or multiple race"]
    rtot = sum(frame["nonhisp_race"][r] for r in resid)
    for r in resid:
        out[r] += cases_by_group["NH Other or multiple"] * frame["nonhisp_race"][r] / rtot
    if mode == "omb_allocated":
        htot = sum(frame["hisp_race"].values())
        for r in RACE6:
            out[r] += hisp_cases * frame["hisp_race"][r] / htot
    elif mode == "hispanic_as_white":
        out["White"] += hisp_cases
    elif mode != "nonhispanic_only":
        raise ValueError(mode)
    return out


def fetch_acs(cache):
    """Read the cached ACS frame (DATA-04)."""
    return json.load(open(cache))


ROWS = []


def emit(stratification, cases, scenario, ses=None):
    tot = float(sum(cases.values()))
    for cat in cases:
        se = (ses or {}).get(cat, np.nan)
        share = cases[cat] / tot
        ROWS.append({"stratification": stratification, "scenario": scenario, "category": cat,
                     "share": share, "relse_pct": round(100 * se / share, 2) if np.isfinite(se) else np.nan})


# definition flag, weight, scenario label, race frames built
PLAN = [
    ("dm_primary", "WTMECPRP", "primary", ("omb_allocated", "nonhispanic_only", "hispanic_as_white")),
    ("dm_a1c_only", "WTMECPRP", "definition_hba1c_only", ("omb_allocated",)),
    ("dm_selfreport_only", "WTMECPRP", "definition_self_report_only", ("omb_allocated",)),
    ("dm_fpg_composite", "WTSAFPRP", "definition_fasting_glucose_composite", ("omb_allocated",)),
    ("dm_type1_excluded", "WTMECPRP", "definition_type1_excluded", ("omb_allocated",)),
]


# ==================================================================================
# Errors of the ALLOCATED six-category race shares. The published widened intervals
# (Table S2) add the relative standard errors of the five NHANES survey-group case shares.
# The shares actually used as PPR denominators are allocated six-category shares. Each is
# linearized as s_r = sum_g a_rg C_g / C, with the ACS allocation fractions a_rg held fixed,
# and its variance estimated under the NHANES stratified PSU design. ACS sampling error and
# error in the allocation assumption are not included in either error set.
# ==================================================================================
def _load_code01():
    import importlib.util
    spec = importlib.util.spec_from_file_location("c1", os.path.join(HERE, "CODE-01-reproduce.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def design_var(strata, psu, u):
    V = 0.0
    df = pd.DataFrame({"s": strata, "p": psu, "u": u})
    for _, g in df.groupby("s"):
        tot = g.groupby("p").u.sum().values
        m = len(tot)
        if m >= 2:
            V += m / (m - 1) * ((tot - tot.mean()) ** 2).sum()
    return V


def allocated_share_errors(d, frame):
    """Survey-design errors of the allocated six-category race shares (RESULTS-12) and the
    widened intervals under both error sets (RESULTS-11)."""
    C1 = _load_code01()
    flag, w = "dm_primary", "WTMECPRP"
    obs = d[flag].notna() & d[w].notna() & (d[w] > 0)
    dd = d[obs]
    y = dd[flag].astype(float).values
    ww = dd[w].values
    grp = dd.RIDRETH3.map(NHANES_GROUP).values
    groups = list(NHANES_GROUP.values())
    groups = list(dict.fromkeys(groups))
    # allocation matrix A[r, g] from the (linear) CODE-02 allocator applied to unit vectors
    A = pd.DataFrame(0.0, index=RACE6, columns=groups)
    for g in groups:
        unit = pd.Series({k: (1.0 if k == g else 0.0) for k in groups})
        alloc = allocate_to_omb_race(unit, frame, "omb_allocated")
        for r in RACE6:
            A.loc[r, g] = alloc[r]
    Cg = pd.Series({g: (ww * y * (grp == g)).sum() for g in groups})
    Ct = (ww * y).sum()
    rows = []
    for r in RACE6:
        a = A.loc[r]
        s = float((a * Cg).sum() / Ct)
        ai = np.array([a[g] for g in grp])
        u = ww * y * (ai - s) / Ct
        se = np.sqrt(design_var(dd.SDMVSTRA.values, dd.SDMVPSU.values, u))
        rows.append({"category": r, "allocated_share": s, "rse_pct": 100 * se / s})
    out = pd.DataFrame(rows)
    # widened intervals under both error sets
    D = pd.read_csv(os.path.join(HERE, "DATA-02-denominators.csv"))
    d2 = D[(D.stratification == "race") & (D.scenario == "primary")].set_index("category").share
    T = pd.read_csv(os.path.join(HERE, "DATA-01-trials.csv"), keep_default_na=False, na_values=[""],
                    dtype={"site_countries": str}).set_index("nct_id")
    T["site_countries"] = T.site_countries.fillna("")
    for c in C1.RACE:
        T[c] = pd.to_numeric(T[c], errors="coerce").fillna(0)
    T["stratum"] = [C1.stratum(c) for c in T.site_countries]
    R = T[T.in_race_cohort == 1].sort_values("race_order")
    survey = {"White": 5.09, "Black or African American": 14.96, "Asian": 16.24}
    alloc_rse = out.set_index("category").rse_pct
    res = []
    for lab, sub in [("All trials", R), ("US sites only", R[R.stratum == "US only"])]:
        est = C1.estimate(sub, C1.RACE)
        for cat, col in [("White", "n_white"), ("Black or African American", "n_black"), ("Asian", "n_asian")]:
            share = float(d2[cat])
            r1, l1, h1 = C1.widen(est[col], share, survey[cat] / 100)
            r2, l2, h2 = C1.widen(est[col], share, alloc_rse[cat] / 100)
            res.append({"cohort": lab, "group": cat, "ppr": r1,
                        "survey_group_rse_pct": survey[cat], "lo_survey": l1, "hi_survey": h1,
                        "allocated_rse_pct": float(alloc_rse[cat]), "lo_allocated": l2, "hi_allocated": h2,
                        "max_abs_limit_change": max(abs(l1 - l2), abs(h1 - h2))})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(HERE, "RESULTS-11-widened-intervals-both-error-sets.csv"), index=False)
    # Rounded to 10 significant digits: the last floating-point digits otherwise vary by platform.
    out.to_csv(os.path.join(HERE, "RESULTS-12-allocated-share-rse.csv"), index=False, float_format="%.10g")
    print(out.round(4).to_string(index=False)); print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    frame = build_population_frame(fetch_acs(os.path.join(HERE, "DATA-04-acs-adult20-frame.json")))
    os.makedirs("nhanes", exist_ok=True)
    for f in ("P_DIQ", "P_GLU"):
        if not os.path.exists(f"nhanes/{f}.xpt"):
            urllib.request.urlretrieve(f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{f}.xpt", f"nhanes/{f}.xpt")
    d = define_conditions(load_nhanes(os.path.join(HERE, "DATA-03-nhanes-person-level.parquet"),
                                      "nhanes/P_DIQ.xpt", "nhanes/P_GLU.xpt"))
    for flag, wt, scen, modes in PLAN:
        cc = nhanes_case_counts(d, flag, wt)
        for mode in modes:
            sc = scen if mode == "omb_allocated" else f"{scen}__race_frame_{mode}"
            emit("race", allocate_to_omb_race(cc["cases_by_group"], frame, mode), sc)
        emit("ethnicity", cc["cases_by_eth"].to_dict(), scen, ses=cc["se_share_by_eth"].to_dict())
        if scen == "primary":
            # five mutually exclusive survey groups; no allocation assumption
            g, se = cc["cases_by_group"], cc["se_share_by_group"]
            ce = {"NH White": g["NH White"], "NH Black": g["NH Black"], "NH Asian": g["NH Asian"],
                  "NH Other or multiple": g["NH Other or multiple"],
                  "Hispanic or Latino": g["Mexican American"] + g["Other Hispanic"]}
            ces = {"NH White": se["NH White"], "NH Black": se["NH Black"], "NH Asian": se["NH Asian"],
                   "NH Other or multiple": se["NH Other or multiple"],
                   "Hispanic or Latino": cc["se_share_by_eth"]["Hispanic or Latino"]}
            emit("race_ethnicity", ce, scen, ses=ces)
    out = pd.DataFrame(ROWS)
    out.loc[out.stratification == "race", "relse_pct"] = np.nan
    out.to_csv(os.path.join(HERE, "RESULTS-10-benchmarks.csv"), index=False)

    ref = pd.read_csv(os.path.join(HERE, "DATA-02-denominators.csv"))
    key = ["stratification", "scenario", "category"]
    m = ref.merge(out, on=key, how="outer", suffixes=("_data02", "_rebuilt"), indicator=True)
    ok = ((m._merge == "both").all()
          and np.allclose(m.share_data02, m.share_rebuilt, rtol=0, atol=1e-12)
          and ((m.relse_pct_data02 - m.relse_pct_rebuilt).abs().fillna(0) < 1e-9).all()
          and (m.relse_pct_data02.isna() == m.relse_pct_rebuilt.isna()).all())
    print(f"{len(out)} benchmark rows; DATA-02 reproduced: {ok}")
    allocated_share_errors(d, frame)
    sys.exit(0 if ok else 1)
