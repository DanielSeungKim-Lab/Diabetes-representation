"""Reproduce every published estimate in:

  Site geography and racial representation in type 2 diabetes trials.

Run:  python CODE-01-reproduce.py
Needs: pandas, numpy, scipy.  Writes RESULTS-01 to RESULTS-09 and RESULTS-13 to
RESULTS-17 beside this script.

Inputs are DATA-01, DATA-02 and DATA-06 to DATA-09. Nothing is fetched from the
network, so the deposit reproduces the paper offline and the registry snapshot
cannot drift underneath it.

Method notes that matter for exact reproduction:
  * Intervals are log-scale Wald intervals on a cluster bootstrap SD, clustering on
    the trial, NOT percentile bootstrap intervals.
  * The bootstrap seed is derived from cohort CONTENT (crc32 of the pipe-joined
    sorted NCT identifiers), so any stratum reproduces when run in isolation and
    stratum order does not affect the result.
  * Where any draw contains no participants of a group the log-scale estimator is
    undefined and the interval is reported as not estimable (blank).
"""
import os, zlib
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 20_000
RACE = ["n_white", "n_black", "n_asian", "n_aian", "n_nhopi", "n_other_multiple"]
ETH = ["n_hispanic", "n_not_hispanic"]
DEN_RACE = {"White": "n_white", "Black or African American": "n_black", "Asian": "n_asian",
            "American Indian or Alaska Native": "n_aian",
            "Native Hawaiian or Other Pacific Islander": "n_nhopi",
            "Other or multiple race": "n_other_multiple"}
DEN_ETH = {"Hispanic or Latino": "n_hispanic", "Not Hispanic or Latino": "n_not_hispanic"}
TERRITORIES = {"Puerto Rico", "Guam", "U.S. Virgin Islands", "American Samoa",
               "Northern Mariana Islands"}
ORDERED = ["US only", "US plus 1 to 4", "US plus 5 to 9", "US plus 10 or more"]
REPORT = ["n_white", "n_black", "n_asian"]
INFLUENTIAL = "NCT01633177"


def cohort_seed(index):
    return zlib.crc32("|".join(sorted(map(str, index))).encode()) % (2 ** 32)


def estimate(sub, groups, n_boot=N_BOOT):
    """Point shares plus log-scale bootstrap SD, clustering on the trial."""
    rng = np.random.default_rng(cohort_seed(sub.index))
    vals = sub[groups].to_numpy(float)
    idx = np.arange(len(sub))
    total = vals.sum()
    point = {g: vals[:, k].sum() / total for k, g in enumerate(groups)}
    draws = np.empty((n_boot, len(groups)))
    kept = 0
    for _ in range(n_boot):
        s = vals[rng.choice(idx, len(idx), replace=True)]
        t = s.sum()
        if t == 0:
            continue
        draws[kept] = s.sum(axis=0) / t
        kept += 1
    draws = draws[:kept]
    return {g: dict(point=point[g],
                    sd=float(np.std(np.log(np.clip(draws[:, k], 1e-9, None)), ddof=1)),
                    zero_draws=int((draws[:, k] == 0).sum()),
                    n_trials=len(sub), participants=int(total))
            for k, g in enumerate(groups)}


def ratio(est_g, share):
    r = est_g["point"] / share
    if est_g["zero_draws"] > 0:
        return r, np.nan, np.nan
    return r, r * np.exp(-1.96 * est_g["sd"]), r * np.exp(1.96 * est_g["sd"])


def widen(est_g, share, relse):
    """Denominator uncertainty added in quadrature on the log scale.

    Independence between the trial-composition estimate and the NHANES prevalence
    estimate is assumed (disjoint data sources), so the covariance term is zero.
    American Community Survey uncertainty is NOT included.
    """
    r = est_g["point"] / share
    sd = float(np.hypot(est_g["sd"], relse))
    return r, r * np.exp(-1.96 * sd), r * np.exp(1.96 * sd)


def stratum(countries, territories_domestic=False):
    """Site stratum. The primary basis counts US territories as FOREIGN.

    An empty site_countries field means the trial posted no locations at all, which is
    a distinct arm from "no US site". Read DATA-01 with keep_default_na=False so the
    empty string does not arrive here as the float NaN; this guard is the second line
    of defense, because NaN would otherwise stringify to "nan" and read as a country.
    """
    if countries is None or (isinstance(countries, float) and np.isnan(countries)):
        countries = ""
    cs = {c for c in str(countries).strip().split(";") if c and c.lower() != "nan"}
    if not cs:
        return "No location posted"
    if "United States" not in cs:
        return "No US site"
    other = cs - {"United States"}
    if territories_domestic:
        other = other - TERRITORIES
    k = len(other)
    if k == 0:
        return "US only"
    return "US plus 1 to 4" if k <= 4 else ("US plus 5 to 9" if k <= 9 else "US plus 10 or more")


def jonckheere_terpstra(groups):
    """Two-sided JT test, normal approximation with the tie-corrected variance.

    Ties contribute 0.5 to U. SciPy carries no JT routine, hence this implementation.
    """
    ks = [np.asarray(g, float) for g in groups]
    U = 0.0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i][:, None], ks[j][None, :]
            U += float((a < b).sum()) + 0.5 * float((a == b).sum())
    ns = np.array([len(g) for g in ks], float)
    N = ns.sum()
    mu = (N ** 2 - (ns ** 2).sum()) / 4.0
    _, tc = np.unique(np.concatenate(ks), return_counts=True)
    var = (N * (N - 1) * (2 * N + 5)
           - (ns * (ns - 1) * (2 * ns + 5)).sum()
           - (tc * (tc - 1) * (2 * tc + 5)).sum()) / 72.0
    var += ((ns * (ns - 1) * (ns - 2)).sum() * (tc * (tc - 1) * (tc - 2)).sum()) \
        / (36.0 * N * (N - 1) * (N - 2))
    var += ((ns * (ns - 1)).sum() * (tc * (tc - 1)).sum()) / (8.0 * N * (N - 1))
    z = (U - mu) / np.sqrt(var)
    return U, z, 2 * stats.norm.sf(abs(z))


def fmt(r, lo, hi):
    if not np.isfinite(lo):
        return "%.2f (not estimable)" % r
    return "%.2f (%.2f-%.2f)" % (r, lo, hi)


def row(sub, groups, shares, label, keys):
    """One output row. A stratum with no reported participants yields blanks, not zeros."""
    total = int(sub[groups].to_numpy(float).sum()) if len(sub) else 0
    out = {"row": label, "trials": len(sub), "participants": total}
    if total == 0:
        for k in keys:
            out[k] = ""
        return out
    est = estimate(sub, groups)
    for k in keys:
        out[k] = fmt(*ratio(est[k], shares[k]))
    return out


def main():
    T = pd.read_csv(os.path.join(HERE, "DATA-01-trials.csv"),
                keep_default_na=False, na_values=[""],
                dtype={"site_countries": str}).set_index("nct_id")
    T["site_countries"] = T.site_countries.fillna("")
    T["enrollment"] = pd.to_numeric(T.enrollment, errors="coerce")
    for c in RACE + ETH:
        T[c] = pd.to_numeric(T[c], errors="coerce").fillna(0)
    D = pd.read_csv(os.path.join(HERE, "DATA-02-denominators.csv"))
    T["stratum"] = [stratum(c) for c in T.site_countries]
    T["n_foreign"] = [len({c for c in str(s).split(";") if c} - {"United States"})
                      for s in T.site_countries]

    def shares(strat, scen, mapping):
        d = D[(D.stratification == strat) & (D.scenario == scen)]
        return {mapping[c]: float(v) for c, v in zip(d.category, d.share) if c in mapping}

    SH = shares("race", "primary", DEN_RACE)
    SE = shares("ethnicity", "primary", DEN_ETH)
    # Cohort order is part of the reproduction contract: the cluster bootstrap resamples
    # trials by position, so each cohort is restored to its own build order before any
    # estimate is taken. Point estimates are sums and are order-invariant; interval
    # bounds move in the second decimal place if the order changes.
    R = T[T.in_race_cohort == 1].sort_values("race_order").copy()
    E = T[T.in_ethnicity_cohort == 1].sort_values("ethnicity_order").copy()

    # ---- Table 2 upper block and the Results arms -------------------------------
    arms = [("All trials", R),
            ("US sites only", R[R.stratum == "US only"]),
            ("Any non-US site", R[R.stratum.isin(["US plus 1 to 4", "US plus 5 to 9",
                                                  "US plus 10 or more", "No US site"])]),
            ("No locations posted", R[R.stratum == "No location posted"])]
    pd.DataFrame([row(s, RACE, SH, lab, REPORT) for lab, s in arms]).to_csv(
        os.path.join(HERE, "RESULTS-01-arms.csv"), index=False)

    # ---- Table 1 trial characteristics (current manuscript layout) -----------------
    # Phase and start date come from the registry frame frozen on 2026-09-25 (DATA-06).
    C6 = pd.read_csv(os.path.join(HERE, "DATA-06-trial-characteristics.csv"),
                     keep_default_na=False, na_values=[""]).set_index("nct_id")
    groups_t1 = [("US-only sites", R[R.stratum == "US only"]),
                 ("Multinational or non-US", R[R.stratum.isin(["US plus 1 to 4", "US plus 5 to 9",
                                                             "US plus 10 or more", "No US site"])]),
                 ("No locations posted", R[R.stratum == "No location posted"])]
    t1 = []
    for lab, s in groups_t1:
        enr = s.enrollment.dropna()
        named = s[RACE].to_numpy(float).sum(axis=1)
        sy = C6.loc[s.index, "start_year"].astype(float)
        ph4 = C6.loc[s.index, "phase"].str.contains("PHASE4")
        ncty = [len({c for c in str(x).split(";") if c}) for x in s.site_countries]
        t1.append({"site_stratum": lab, "trials": len(s),
                   "trials_contributing_race": int((named > 0).sum()),
                   "participants_known_race": int(named.sum()),
                   "median_enrollment": float(enr.median()),
                   "enrollment_q1": float(enr.quantile(.25)), "enrollment_q3": float(enr.quantile(.75)),
                   "median_start_year": float(sy.median()),
                   "start_year_q1": float(sy.quantile(.25)), "start_year_q3": float(sy.quantile(.75)),
                   "industry_n": int((s.lead_sponsor_class == "INDUSTRY").sum()),
                   "industry_pct": round(100 * (s.lead_sponsor_class == "INDUSTRY").mean(), 1),
                   "median_countries_listed": (float(np.median(ncty)) if lab != "No locations posted" else ""),
                   "phase4_n": int(ph4.sum()), "phase4_pct": round(100 * ph4.mean(), 1)})
    pd.DataFrame(t1).to_csv(os.path.join(HERE, "RESULTS-02-table1.csv"), index=False)


    # ---- Table 2 gradient and Figure 1B ------------------------------------------
    grad = [row(R[R.stratum == st], RACE, SH, st, REPORT) for st in ORDERED]
    grad += [row(R[R.stratum == "No US site"], RACE, SH, "No US site", REPORT),
             row(R[R.stratum == "No location posted"], RACE, SH, "No locations posted", REPORT)]
    pd.DataFrame(grad).to_csv(os.path.join(HERE, "RESULTS-03-gradient.csv"), index=False)

    # ---- Table 2 trend row --------------------------------------------------------
    trend = []
    named = R[RACE].to_numpy(float).sum(axis=1)
    for g in REPORT:
        sh = np.where(named > 0, R[g].to_numpy(float) / np.where(named > 0, named, 1), np.nan)
        S = pd.Series(sh, index=R.index)
        gs = [S[R.stratum == st].dropna().to_numpy() for st in ORDERED]
        U, z, p = jonckheere_terpstra(gs)
        lab = np.concatenate([[i] * len(x) for i, x in enumerate(gs)])
        tau = stats.kendalltau(lab, np.concatenate(gs), variant="b")
        trend.append({"group": g, "trials": int(sum(len(x) for x in gs)),
                      "participants": int(R[R.stratum.isin(ORDERED)][RACE].to_numpy(float).sum()),
                      "medians": "; ".join("%.1f" % (100 * np.median(x)) for x in gs),
                      "z": round(float(z), 2), "tau_b": round(float(tau.statistic), 3),
                      "p_value": "%.1e" % p})
    pd.DataFrame(trend).to_csv(os.path.join(HERE, "RESULTS-04-trend-tests.csv"), index=False)

    # ---- Table 3 within US-only trials ---------------------------------------------
    U = R[R.stratum == "US only"].copy()
    q1, q2 = U.enrollment.quantile(1 / 3), U.enrollment.quantile(2 / 3)
    U["tertile"] = np.where(U.enrollment <= q1, 1, np.where(U.enrollment <= q2, 2, 3))
    t3 = [row(U, RACE, SH, "All US-only trials", REPORT)]
    for t in (1, 2, 3):
        s = U[U.tertile == t]
        t3.append(row(s, RACE, SH, "Enrollment tertile %d (%d-%d)"
                      % (t, s.enrollment.min(), s.enrollment.max()), REPORT))
    t3.append(row(U[U.lead_sponsor_class == "INDUSTRY"], RACE, SH, "Industry sponsor", REPORT))
    t3.append(row(U[U.lead_sponsor_class != "INDUSTRY"], RACE, SH, "Non-industry sponsor", REPORT))
    if INFLUENTIAL in U.index:
        t3.append(row(U.drop(index=INFLUENTIAL), RACE, SH,
                      "All US-only trials, excluding %s" % INFLUENTIAL, REPORT))
    pd.DataFrame(t3).to_csv(os.path.join(HERE, "RESULTS-05-table3-within-us.csv"), index=False)

    # ---- Table S1 sensitivity to case definition and racial classification ----------
    scen = [("Primary", "race", "primary"),
            ("Self-reported diagnosis only", "race", "definition_self_report_only"),
            ("HbA1c only", "race", "definition_hba1c_only"),
            ("Fasting glucose composite", "race", "definition_fasting_glucose_composite"),
            ("Likely type 1 diabetes excluded", "race", "definition_type1_excluded"),
            ("Hispanic counted as White", "race", "primary__race_frame_hispanic_as_white"),
            ("Non-Hispanic frame", "race", "primary__race_frame_nonhispanic_only")]
    s1 = []
    for lab, strat, sc in scen:
        sh = shares(strat, sc, DEN_RACE)
        if len(sh) < len(DEN_RACE):
            continue
        s1.append(row(U, RACE, sh, lab, REPORT))
    pd.DataFrame(s1).to_csv(os.path.join(HERE, "RESULTS-06-sensitivity.csv"), index=False)

    # ---- Table S2 denominator uncertainty -------------------------------------------
    rel = D[(D.stratification == "race_ethnicity") & (D.scenario == "primary")]
    RELSE = {"n_white": "NH White", "n_black": "NH Black", "n_asian": "NH Asian"}
    relmap = {g: float(rel.loc[rel.category == c, "relse_pct"].iloc[0]) / 100
              for g, c in RELSE.items()}
    s2 = []
    for lab, s in (("All trials", R), ("US sites only", U)):
        est = estimate(s, RACE)
        for g in REPORT:
            r, lo, hi = ratio(est[g], SH[g])
            wr, wlo, whi = widen(est[g], SH[g], relmap[g])
            s2.append({"group": g, "arm": lab, "ppr": round(r, 2),
                       "trial_only_ci": fmt(r, lo, hi), "widened_ci": fmt(wr, wlo, whi),
                       "denominator_relse_pct": round(100 * relmap[g], 1)})
    estE = {lab: estimate(s, ETH) for lab, s in
            (("All trials", E), ("US sites only", E[E.stratum == "US only"]))}
    relH = float(rel.loc[rel.category == "Hispanic or Latino", "relse_pct"].iloc[0]) / 100
    for lab, est in estE.items():
        r, lo, hi = ratio(est["n_hispanic"], SE["n_hispanic"])
        wr, wlo, whi = widen(est["n_hispanic"], SE["n_hispanic"], relH)
        s2.append({"group": "n_hispanic", "arm": lab, "ppr": round(r, 2),
                   "trial_only_ci": fmt(r, lo, hi), "widened_ci": fmt(wr, wlo, whi),
                   "denominator_relse_pct": round(100 * relH, 1)})
    pd.DataFrame(s2).to_csv(os.path.join(HERE, "RESULTS-07-denominator-uncertainty.csv"), index=False)

    # ---- Hispanic/Latinx ethnicity axis ----------------------------------------------
    earms = [("All trials", E),
             ("US sites only", E[E.stratum == "US only"]),
             ("Any non-US site", E[E.stratum.isin(ORDERED[1:] + ["No US site"])]),
             ("No US site", E[E.stratum == "No US site"]),
             ("No locations posted", E[E.stratum == "No location posted"])]
    e_out = [row(s, ETH, SE, lab, ["n_hispanic", "n_not_hispanic"]) for lab, s in earms if len(s)]
    eg = [E[E.stratum == st] for st in ORDERED]
    namedE = E[ETH].to_numpy(float).sum(axis=1)
    SH_E = pd.Series(np.where(namedE > 0, E.n_hispanic.to_numpy(float) / np.where(namedE > 0, namedE, 1), np.nan),
                     index=E.index)
    gsE = [SH_E[E.stratum == st].dropna().to_numpy() for st in ORDERED]
    _, zE, pE = jonckheere_terpstra(gsE)
    labE = np.concatenate([[i] * len(x) for i, x in enumerate(gsE)])
    tauE = stats.kendalltau(labE, np.concatenate(gsE), variant="b")
    e_out.append({"row": "Trend across the four US strata",
                  "trials": int(sum(len(x) for x in gsE)),
                  "participants": int(pd.concat(eg)[ETH].to_numpy(float).sum()),
                  "n_hispanic": "z = %+.2f, tau-b = %.3f, P = %.2f" % (zE, tauE.statistic, pE),
                  "n_not_hispanic": ""})
    pd.DataFrame(e_out).to_csv(os.path.join(HERE, "RESULTS-08-ethnicity.csv"), index=False)


    # ---- Supplemental Table S3, US territories counted as domestic --------------------
    # The primary basis treats a US territory as a site outside the United States. This
    # table repeats the geographic strata with territories grouped alongside the 50 states
    # and the District of Columbia, which moves trials into the US-containing strata.
    T["stratum_terr"] = [stratum(c, territories_domestic=True) for c in T.site_countries]
    Rt = T[T.in_race_cohort == 1].sort_values("race_order").copy()
    LBL = {"US only": "US or US-territory sites only",
           "US plus 1 to 4": "Plus 1-4 other countries",
           "US plus 5 to 9": "Plus 5-9 other countries",
           "US plus 10 or more": "Plus 10 or more other countries",
           "No US site": "No US or US-territory site",
           "No location posted": "No locations posted"}
    s3 = [row(Rt[Rt.stratum_terr == st], RACE, SH, LBL[st], REPORT)
          for st in ORDERED + ["No US site", "No location posted"]]
    pd.DataFrame(s3).to_csv(os.path.join(HERE, "RESULTS-09-territories.csv"), index=False)

    print("race cohort      %d trials, %d participants"
          % (len(R), int(R[RACE].to_numpy(float).sum())))
    print("ethnicity cohort %d trials, %d participants"
          % (len(E), int(E[ETH].to_numpy(float).sum())))
    print("enrollment tertile cut points %.1f and %.1f" % (q1, q2))
    print("wrote RESULTS-01 to RESULTS-09")


# ==================================================================================
# Asian enrollment by whether a trial listed a site in Asia
# (RESULTS-13, RESULTS-14). A trial has "a site in Asia" when its registered locations
# include at least one country in East, South, or Southeast Asia, the regions of origin
# named in the US federal definition of the Asian race category. ASIA lists every such
# country in DATA-01; the run stops if another one appears. Site lists are trial-level,
# so the comparison is ecological.
# ==================================================================================
ASIA = {"China", "Hong Kong", "India", "Japan", "Malaysia", "Philippines", "Singapore",
        "South Korea", "Sri Lanka", "Taiwan", "Thailand", "Vietnam"}
# Countries in the same regions that would need adding if they ever appeared in the data.
ASIA_NOT_PRESENT = {"Bangladesh", "Pakistan", "Nepal", "Bhutan", "Maldives", "Indonesia", "Cambodia",
                    "Laos", "Lao People's Democratic Republic", "Myanmar", "Brunei", "Brunei Darussalam",
                    "Timor-Leste", "Mongolia", "North Korea", "Macau", "Macao", "Korea, Republic of",
                    "Viet Nam"}
GROUPS_ASIA = {"Asian": "n_asian", "Black": "n_black", "White": "n_white"}


def asia_sites():
    T = pd.read_csv(os.path.join(HERE, "DATA-01-trials.csv"), keep_default_na=False, na_values=[""],
                    dtype={"site_countries": str}).set_index("nct_id")
    T["site_countries"] = T.site_countries.fillna("")
    for col in RACE:
        T[col] = pd.to_numeric(T[col], errors="coerce").fillna(0)
    D = pd.read_csv(os.path.join(HERE, "DATA-02-denominators.csv"))
    d = D[(D.stratification == "race") & (D.scenario == "primary")]
    SH = {DEN_RACE[k]: float(v) for k, v in zip(d.category, d.share) if k in DEN_RACE}

    R = T[T.in_race_cohort == 1].sort_values("race_order").copy()
    R["stratum"] = [stratum(s) for s in R.site_countries]
    R["cs"] = [{x for x in s.split(";") if x} for s in R.site_countries]
    seen = set().union(*R.cs)
    assert not (seen & ASIA_NOT_PRESENT), sorted(seen & ASIA_NOT_PRESENT)
    R["any_asia"] = [bool(s & ASIA) for s in R.cs]
    R["all_asia"] = [bool(s) and s <= ASIA for s in R.cs]
    R["named"] = R[RACE].sum(axis=1)

    located = R[R.stratum != "No location posted"]
    us_only = R[R.stratum == "US only"]
    us_foreign = R[R.stratum.isin(["US plus 1 to 4", "US plus 5 to 9", "US plus 10 or more"])]
    no_us = R[R.stratum == "No US site"]
    strata = [
        ("All trials", R),
        ("Trials with posted locations", located),
        ("Posted locations, any site in Asia", located[located.any_asia]),
        ("Posted locations, no site in Asia", located[~located.any_asia]),
        ("US sites only", us_only),
        ("US plus foreign sites, any site in Asia", us_foreign[us_foreign.any_asia]),
        ("US plus foreign sites, no site in Asia", us_foreign[~us_foreign.any_asia]),
        ("No US site, all sites in Asia", no_us[no_us.all_asia]),
        ("No US site, sites in Asia and elsewhere", no_us[no_us.any_asia & ~no_us.all_asia]),
        ("No US site, no site in Asia", no_us[~no_us.any_asia]),
        ("No locations posted", R[R.stratum == "No location posted"]),
    ]
    rows = []
    for label, sub in strata:
        est = estimate(sub, RACE)
        total = float(sub.named.sum())
        for g, col in GROUPS_ASIA.items():
            r, lo, hi = ratio(est[col], SH[col])
            rows.append({"stratum": label, "group": g, "trials": len(sub), "participants_known_race": int(total),
                         "group_participants": int(sub[col].sum()),
                         "group_share_pct": round(100 * sub[col].sum() / total, 2),
                         "us_case_share_pct": round(100 * SH[col], 2),
                         "ppr": round(r, 2),
                         "ci_low": None if np.isnan(lo) else round(lo, 2),
                         "ci_high": None if np.isnan(hi) else round(hi, 2),
                         "bootstrap_draws_with_zero": est[col]["zero_draws"]})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "RESULTS-13-asia-sites.csv"), index=False)

    asian_all = float(R.n_asian.sum())
    asian_loc = float(located.n_asian.sum())
    with_asia = located[located.any_asia]
    known_us = us_only[us_only.named > 0]
    single = no_us[no_us.cs.map(len) == 1]
    desc = [
        ("Asian participants, all trials", int(asian_all)),
        ("Asian participants as % of all participants with known race", round(100 * asian_all / R.named.sum(), 1)),
        ("Black participants, all trials", int(R.n_black.sum())),
        ("Black participants as % of all participants with known race", round(100 * R.n_black.sum() / R.named.sum(), 1)),
        ("Asian participants, US-only trials", int(us_only.n_asian.sum())),
        ("Asian participants as % of US-only participants", round(100 * us_only.n_asian.sum() / us_only.named.sum(), 1)),
        ("Black participants, US-only trials", int(us_only.n_black.sum())),
        ("Black participants as % of US-only participants", round(100 * us_only.n_black.sum() / us_only.named.sum(), 1)),
        ("US-only trials with known race", len(known_us)),
        ("US-only trials with known race and no Asian participants", int((known_us.n_asian == 0).sum())),
        ("Asian participants in trials with posted locations", int(asian_loc)),
        ("Asian participants in trials with any site in Asia", int(with_asia.n_asian.sum())),
        ("... as % of Asian participants in trials with posted locations", round(100 * with_asia.n_asian.sum() / asian_loc, 1)),
        ("... as % of all Asian participants", round(100 * with_asia.n_asian.sum() / asian_all, 1)),
        ("US-only share of all participants, %", round(100 * us_only.named.sum() / R.named.sum(), 1)),
        ("US-only share of all Black participants, %", round(100 * us_only.n_black.sum() / R.n_black.sum(), 1)),
        ("US-only share of all Asian participants, %", round(100 * us_only.n_asian.sum() / asian_all, 1)),
        ("No-US-site trials conducted in a single country", len(single)),
    ]
    desc += [(f"No-US-site trials conducted in {k} alone", int(v))
             for k, v in single.cs.map(lambda s: next(iter(s))).value_counts().items() if k in ASIA]
    desc += [(f"Trials listing a site in {k}", int(sum(k in s for s in R.cs))) for k in sorted(ASIA)]
    desc = [(q, f"{v:.1f}" if isinstance(v, float) else str(int(v))) for q, v in desc]
    pd.DataFrame(desc, columns=["quantity", "value"]).to_csv(
        os.path.join(HERE, "RESULTS-14-asia-site-countries.csv"), index=False)


# ==================================================================================
# Other post hoc restrictions (RESULTS-15), US-only facility records by state against
# ACS state population (RESULTS-16, from DATA-07 and DATA-08), and every box of the
# Figure 1 flow (RESULTS-17, from DATA-09 and DATA-01).
# ==================================================================================
GROUPS_POSTHOC = {"White": "n_white", "Black": "n_black", "Asian": "n_asian"}


def load_posthoc():
    T = pd.read_csv(os.path.join(HERE, "DATA-01-trials.csv"), keep_default_na=False, na_values=[""],
                    dtype={"site_countries": str}).set_index("nct_id")
    T["site_countries"] = T.site_countries.fillna("")
    for col in RACE + ETH:
        T[col] = pd.to_numeric(T[col], errors="coerce").fillna(0)
    T["stratum"] = [stratum(s) for s in T.site_countries]
    D = pd.read_csv(os.path.join(HERE, "DATA-02-denominators.csv"))
    d = D[(D.stratification == "race") & (D.scenario == "primary")]
    SH = {DEN_RACE[k]: float(v) for k, v in zip(d.category, d.share) if k in DEN_RACE}
    C6 = pd.read_csv(os.path.join(HERE, "DATA-06-trial-characteristics.csv"),
                     keep_default_na=False, na_values=[""]).set_index("nct_id")
    return T, SH, C6


def posthoc(T, SH, C6):
    R = T[T.in_race_cohort == 1].sort_values("race_order").copy()
    R["minimum_age_years"] = C6.loc[R.index, "minimum_age_years"]
    cuts = [("Primary analysis", R),
            ("Dedicated race tables only (race_scheme = OMB race)", R[R.race_scheme == "OMB race"]),
            ("Excluding trials allowing enrollment before age 18 years", R[~(R.minimum_age_years < 18)])]
    rows = []
    for label, sub in cuts:
        for geo, s in [("All trials", sub), ("US sites only", sub[sub.stratum == "US only"])]:
            est = estimate(s, RACE)
            out = {"analysis": label, "geography": geo, "trials": len(s),
                   "participants_known_race": int(s[RACE].to_numpy(float).sum())}
            for g, col in GROUPS_POSTHOC.items():
                out[g] = fmt(*ratio(est[col], SH[col]))
            rows.append(out)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "RESULTS-15-posthoc.csv"), index=False)
    n_young = int((R.minimum_age_years < 18).sum())
    n_missing_age = int(R.minimum_age_years.isna().sum())
    return n_young, n_missing_age


def sites(T):
    F = pd.read_csv(os.path.join(HERE, "DATA-07-us-only-facilities.csv"), keep_default_na=False, na_values=[""])
    P = pd.read_csv(os.path.join(HERE, "DATA-08-acs-state-population.csv"))
    us_only = set(T.index[(T.stratum == "US only") & (T.in_race_cohort == 1)])
    assert set(F.nct_id) <= us_only, "facility file contains a trial that is not US-only"
    assert (F.country == "United States").all() and F.state.notna().all()
    by = F.groupby("state").size().rename("facility_records")
    out = P.set_index("state").join(by).fillna({"facility_records": 0})
    out["facility_records"] = out.facility_records.astype(int)
    out["facility_share_pct"] = 100 * out.facility_records / out.facility_records.sum()
    out["population_share_pct"] = 100 * out.total_pop / out.total_pop.sum()
    out["black_resident_share_pct"] = 100 * out.black_alone / out.black_alone.sum()
    out["asian_resident_share_pct"] = 100 * out.asian_alone / out.asian_alone.sum()
    out = out.sort_values("facility_records", ascending=False).round(2)
    out.reset_index().to_csv(os.path.join(HERE, "RESULTS-16-us-sites-by-state.csv"), index=False)
    ca_ny = out.loc[["California", "New York"]]
    return int(out.facility_records.sum()), F.nct_id.nunique(), int(ca_ny.facility_records.sum()), \
        round(float(100 * ca_ny.facility_records.sum() / out.facility_records.sum()), 1), \
        round(float(ca_ny.asian_resident_share_pct.sum()), 1)


def flow(T):
    S = pd.read_csv(os.path.join(HERE, "DATA-09-screening-disposition.csv"))
    n = lambda pat: int(S.disposition.str.contains(pat, regex=False).sum())
    R = T[T.in_race_cohort == 1]
    named = R[RACE].to_numpy(float).sum(axis=1)
    arm = lambda m: (int(m.sum()), int(named[m.to_numpy()].sum()))
    us, foreign, noloc = (R.stratum == "US only"), R.stratum.isin(
        ["US plus 1 to 4", "US plus 5 to 9", "US plus 10 or more", "No US site"]), (R.stratum == "No location posted")
    rows = [("Screened (registry query, frozen 2026-09-25)", 615),
            ("Excluded: no baseline characteristics module (not retained in frame)", 615 - len(S)),
            ("Posted a baseline characteristics module", len(S)),
            ("Excluded: no race measure of any kind", n("no race measure")),
            ("  of which race declared not collected", n("declared not collected")),
            ("Posted a race measure of any kind", len(S) - n("no race measure")),
            ("Excluded: labels did not map to six race categories", n("did not map")),
            ("Excluded: posted rows covered under half of enrollment", n("under half")),
            ("Excluded: race rows exceeded enrollment by over 10%", n("exceeded enrollment")),
            ("Excluded: no race counts in the selected scheme", n("no race counts")),
            ("Met the coverage rules", int((S.step >= "5").sum())),
            ("Excluded: registered only for type 1 diabetes", n("type 1 diabetes")),
            ("Analyzed race cohort, trials", int((S.in_race_cohort == 1).sum())),
            ("Analyzed race cohort, participants with known race", int(named.sum())),
            ("US sites only, trials", arm(us)[0]), ("US sites only, participants", arm(us)[1]),
            ("Any non-US site, trials", arm(foreign)[0]), ("Any non-US site, participants", arm(foreign)[1]),
            ("No locations posted, trials", arm(noloc)[0]), ("No locations posted, participants", arm(noloc)[1])]
    out = pd.DataFrame(rows, columns=["flow_box", "count"])
    out.to_csv(os.path.join(HERE, "RESULTS-17-flow-counts.csv"), index=False)
    assert int((S.in_race_cohort == 1).sum()) == len(R), "disposition file and DATA-01 disagree"
    return out


def posthoc_sites_flow():
    T, SH, C6 = load_posthoc()
    n_young, n_missing_age = posthoc(T, SH, C6)
    total, trials, ca_ny, ca_ny_pct, ca_ny_asian = sites(T)
    fl = flow(T)
    print(f"Trials allowing enrollment before 18 years: {n_young} (minimum age missing for {n_missing_age})")
    print(f"US-only facility records: {total} in {trials} trials; California and New York {ca_ny} "
          f"({ca_ny_pct}%) vs {ca_ny_asian}% of US Asian residents")
    print(fl.to_string(index=False))


if __name__ == "__main__":
    main()
    asia_sites()
    posthoc_sites_flow()
