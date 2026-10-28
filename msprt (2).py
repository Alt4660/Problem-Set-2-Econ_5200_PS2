"""
Always-valid sequential testing: Gaussian-mixture SPRT (mSPRT).  numpy + scipy only.

Setup: two arms, unit-variance outcomes (sigma = 1), paired arrival of one
observation per arm.  Per-pair difference d_i ~ N(delta, 2*sigma^2).
H0: delta = 0.  Mixing distribution on delta: N(0, tau2).

    Lambda_n = (1 + tau2*I_n)^(-1/2) * exp( tau2 * U_n^2 / (2*(1 + tau2*I_n)) )
    I_n = n / var_d   (Fisher information),   U_n = sum(d_i) / var_d   (score)

Under H0, Lambda_n is a nonnegative martingale with mean 1, so Ville's
inequality gives  P(sup_n Lambda_n >= 1/alpha) <= alpha  -- for ANY looking
schedule and ANY data-dependent stopping rule.
"""
import numpy as np
from scipy import stats

LOG_COLUMNS = ["is_real", "true_effect", "n_per_arm", "se", "estimate",
               "z", "p_value", "power", "essentially_zero"]


# ----------------------------------------------------------------------------
# tau^2 from the effect scale
# ----------------------------------------------------------------------------
def choose_tau2(true_effect, is_real):
    """tau2 = E[true_effect^2 | real]  (second moment of real effects).

    Why: the mixture likelihood ratio is the Bayes-optimal statistic when the
    mixing prior matches the effects you want to detect.  A zero-mean normal
    prior has no mean, so the right matching quantity is the second moment
    (mean^2 + var), i.e. the RMS effect, not the mean squared.  Too small a
    tau2 wastes power on effects above the prior's scale; too large a tau2
    spreads prior mass over effects that never occur and pays an Occam penalty
    (1 + tau2*I)^(-1/2) at every look.  Computed from the log, not hand-picked.
    """
    te = np.asarray(true_effect, float)[np.asarray(is_real, bool)]
    return float(np.mean(te ** 2))


# ----------------------------------------------------------------------------
# The test
# ----------------------------------------------------------------------------
def msprt(treatment, control, tau2, alpha=0.05, sigma=1.0, looks=None):
    """Always-valid mSPRT over a stream of paired observations.

    treatment, control : 1-D arrays, same length; observation i of each arm
                         arrives together.
    tau2               : mixing variance on the effect (see choose_tau2).
    looks              : optional sorted array of sample sizes (per arm) at
                         which decisions are reported; default = every pair.
                         Validity holds for any schedule.

    Returns dict:
      n             sample size per arm at each look
      estimate      running effect estimate (mean difference)
      log_lambda    log mixture likelihood ratio at each look
      crossed       Lambda_n >= 1/alpha at this look
      reject        stopping decision at each look (True from first crossing on)
      always_valid_p  running-min of 1/Lambda, capped at 1 (valid at any stop)
      stop_at       per-arm n at first crossing, or None
    """
    x = np.asarray(treatment, float)
    y = np.asarray(control, float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("treatment and control must be 1-D arrays of equal length")
    if tau2 <= 0 or not (0 < alpha < 1):
        raise ValueError("need tau2 > 0 and 0 < alpha < 1")

    var_d = 2.0 * sigma ** 2
    n_all = np.arange(1, x.size + 1)
    S = np.cumsum(x - y)
    I = n_all / var_d
    U = S / var_d
    log_lam_all = -0.5 * np.log1p(tau2 * I) + tau2 * U ** 2 / (2.0 * (1.0 + tau2 * I))

    # the running max / p-value must use EVERY observation, even if we only
    # report at a subset of looks; the subset is applied afterwards
    crossed_all = log_lam_all >= -np.log(alpha)
    run_max = np.maximum.accumulate(log_lam_all)
    p_all = np.minimum(1.0, np.exp(-run_max))
    reject_all = np.maximum.accumulate(crossed_all)
    est_all = S / n_all

    if looks is not None:
        idx = np.asarray(looks, int) - 1
        if idx.min() < 0 or idx.max() >= x.size:
            raise ValueError("looks must lie in 1..len(stream)")
    else:
        idx = slice(None)

    stop_at = int(n_all[np.argmax(crossed_all)]) if crossed_all.any() else None
    return dict(n=n_all[idx], estimate=est_all[idx], log_lambda=log_lam_all[idx],
                crossed=crossed_all[idx], reject=reject_all[idx],
                always_valid_p=p_all[idx], stop_at=stop_at)


class OnlineMSPRT:
    """O(1)-state version for production: call update(x_t, y_t) per pair."""

    def __init__(self, tau2, alpha=0.05, sigma=1.0):
        self.tau2, self.alpha, self.var_d = tau2, alpha, 2.0 * sigma ** 2
        self.n, self.S, self.max_loglam, self.stopped_at = 0, 0.0, -np.inf, None

    def update(self, x_t, y_t):
        self.n += 1
        self.S += x_t - y_t
        I, U = self.n / self.var_d, self.S / self.var_d
        ll = -0.5 * np.log1p(self.tau2 * I) + self.tau2 * U ** 2 / (2 * (1 + self.tau2 * I))
        self.max_loglam = max(self.max_loglam, ll)
        if self.stopped_at is None and ll >= -np.log(self.alpha):
            self.stopped_at = self.n
        return dict(n=self.n, log_lambda=ll, reject=self.stopped_at is not None,
                    always_valid_p=min(1.0, float(np.exp(-self.max_loglam))))


# ----------------------------------------------------------------------------
# Log structure + stream construction
# ----------------------------------------------------------------------------
def load_log(path):
    """Load your experiment log (CSV with LOG_COLUMNS)."""
    import csv
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {c: np.array([r[c] for r in rows]) for c in LOG_COLUMNS}
    for c in ("true_effect", "se", "estimate", "z", "p_value", "power"):
        cols[c] = cols[c].astype(float)
    cols["n_per_arm"] = cols["n_per_arm"].astype(int)
    for c in ("is_real", "essentially_zero"):
        cols[c] = np.array([str(v).lower() in ("1", "true") for v in cols[c]])
    return cols


def make_synthetic_log(rng, m=200, frac_real=0.5):
    """Stand-in with the stated structure: n_per_arm in [50, 2000]; real
    effects >= 0.03 with mean ~0.10 (0.03 + Exponential(0.07)); sigma = 1."""
    n = np.exp(rng.uniform(np.log(50), np.log(2000), m)).astype(int)
    is_real = rng.random(m) < frac_real
    te = np.where(is_real, 0.03 + rng.exponential(0.07, m), 0.0)
    se = np.sqrt(2.0 / n)
    est = te + se * rng.standard_normal(m)
    z = est / se
    p = 2 * stats.norm.sf(np.abs(z))
    zc = stats.norm.isf(0.025)
    power = stats.norm.sf(zc - te / se) + stats.norm.cdf(-zc - te / se)
    return dict(is_real=is_real, true_effect=te, n_per_arm=n, se=se, estimate=est,
                z=z, p_value=p, power=power, essentially_zero=~is_real)


def make_stream(n_per_arm, true_effect, rng):
    """One experiment stream: control ~ N(0,1), treatment ~ N(true_effect,1)."""
    return rng.normal(true_effect, 1.0, n_per_arm), rng.normal(0.0, 1.0, n_per_arm)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def wilson(k, n, conf=0.95):
    z = stats.norm.isf((1 - conf) / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def evaluate(log, tau2, n_streams=1000, null=True, alpha=0.05, rng=None, n_override=None):
    """Return rates for mSPRT, naive peek-every-look z-test, and fixed-horizon z-test."""
    rng = rng or np.random.default_rng(0)
    m = len(log["n_per_arm"])
    zc = stats.norm.isf(alpha / 2)
    rej = peek = fixed = 0
    stop_frac = []
    for s in range(n_streams):
        j = rng.integers(m) if null else s % m
        n = int(log["n_per_arm"][j]) if n_override is None else n_override
        te = 0.0 if null else float(log["true_effect"][j])
        x, y = make_stream(n, te, rng)
        out = msprt(x, y, tau2, alpha)
        rej += out["stop_at"] is not None
        if out["stop_at"] is not None:
            stop_frac.append(out["stop_at"] / n)
        zs = np.cumsum(x - y) / np.sqrt(2.0 * np.arange(1, n + 1))
        peek += bool(np.any(np.abs(zs) > zc))
        fixed += bool(abs(zs[-1]) > zc)
    return dict(msprt=rej / n_streams, peek=peek / n_streams, fixed=fixed / n_streams,
                msprt_ci=wilson(rej, n_streams), med_stop_frac=float(np.median(stop_frac)) if stop_frac else np.nan)


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(2026)
    log = load_log(sys.argv[1]) if len(sys.argv) > 1 else make_synthetic_log(rng)
    src = sys.argv[1] if len(sys.argv) > 1 else "SYNTHETIC log (200 rows)"
    tau2 = choose_tau2(log["true_effect"], log["is_real"])
    real = log["is_real"]
    print(f"log: {src};  real effects: n={real.sum()}, min={log['true_effect'][real].min():.3f}, "
          f"mean={log['true_effect'][real].mean():.3f}, RMS={np.sqrt(tau2):.3f}")
    print(f"tau2 = E[effect^2 | real] = {tau2:.5f}  (tau = {np.sqrt(tau2):.3f})\n")

    # sanity: online == vectorised
    x, y = make_stream(500, 0.1, np.random.default_rng(1))
    v = msprt(x, y, tau2)
    o = OnlineMSPRT(tau2)
    ll = np.array([o.update(a, b)["log_lambda"] for a, b in zip(x, y)])
    assert np.allclose(ll, v["log_lambda"]), "online and vectorised disagree"
    print("online == vectorised: OK")

    # 1) pure-null, 1,000 streams, n from log
    r = evaluate(log, tau2, 1000, null=True, rng=np.random.default_rng(11))
    print("\n[1] PURE NULL, 1000 streams, n_per_arm drawn from log, look at every pair")
    print(f"    mSPRT false positive rate : {r['msprt']:.3f}   95% CI [{r['msprt_ci'][0]:.3f}, {r['msprt_ci'][1]:.3f}]")
    print(f"    naive z-test at every look: {r['peek']:.3f}")
    print(f"    fixed-horizon z at end    : {r['fixed']:.3f}   (MC SE ~ 0.007)")

    # 2) horizon sweep (Ville bound is for an INFINITE horizon)
    print("\n[2] mSPRT FPR vs horizon (1000 null streams each)")
    for nmax in (50, 200, 2000, 20000):
        rr = evaluate(log, tau2, 1000, null=True, rng=np.random.default_rng(nmax), n_override=nmax)
        print(f"    n_per_arm = {nmax:>6}: mSPRT {rr['msprt']:.3f}   naive peek {rr['peek']:.3f}")

    # 3) sensitivity to tau2 (null FPR with log-drawn n; power on real rows)
    print("\n[3] tau2 sensitivity   (power = reject rate on the log's real experiments, 1 stream each x20)")
    idx = np.where(real)[0]
    sub = {k: v[idx] for k, v in log.items()}
    for label, t2 in (("0.1x", tau2 / 10), ("1x (chosen)", tau2), ("10x", tau2 * 10), ("hand 1.0", 1.0)):
        rn = evaluate(log, t2, 1000, null=True, rng=np.random.default_rng(5))
        rp = evaluate(sub, t2, 20 * len(idx), null=False, rng=np.random.default_rng(6))
        print(f"    tau2 {label:>11} = {t2:.5f}: null FPR {rn['msprt']:.3f} | power {rp['msprt']:.3f} "
              f"(fixed-horizon {rp['fixed']:.3f}, median stop at {rp['med_stop_frac']:.2f} of n)")
