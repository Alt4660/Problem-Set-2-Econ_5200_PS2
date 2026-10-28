# AI appendix: Task 4.1 (the sequential design)

**Model:** Claude Sonnet 5.5  **Date:** 27 October 2026

How this was made: both chats were exported to PDF and the text copied below. Only the PDF page headers/footers and interface text ("Show more") were removed; the model's wording and code are otherwise unchanged. Nothing the model produced has been summarised or left out.

---

## Version 1

### Prompt v1 (the given prompt, as I first sent it)

My first message in the chat was a short opener (shown with its reply below), then the prompt:

```text
[Prep] Act as an experimentation platform engineer who has implemented sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true effects. Implement an always-valid sequential test -- a mixture sequential probability ratio test or an alpha-spending boundary -- that allows continuous monitoring without inflating the Type I error rate. Give me a function that takes a stream of observations and returns the stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every look inflates Type I error, and precisely what the always-valid construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical false positive rate. If it is not close to 5%, tell me what is wrong.
```

### Raw exchange, v1 (opener, reply, and the model's final answer)

```text
Sequential testing for A/B experiments                                                                            Share
                           Hello Claude, I am about to send you a prompt for my Causal machine leanring
                           course, please prepare.

   Ready when you are. Send over the prompt for your causal machine learning course and
   I’ll get started.
   If it helps me tailor the answer, feel free to include any of these along with it:
          The assignment or problem text, plus any attached data, code, or lecture notes
          The framework your course uses (potential outcomes, structural causal
          models/DAGs, or both)
          Preferred tools (Python with DoWhy, EconML, CausalML, or R)
          The format you need back (written explanation, code, derivation, or a file like a
          notebook or PDF)

[prompt v1 sent here, as above]

Planning the sequential test run.
      I’ll check whether your log structure or portfolio files are in the workspace, then build and
      run the test.

      Looked for files 
      No existing files turned up, so I don’t have your log structure. I’ll use a columnar per-look
      log and keep it easy to remap. I’m writing the implementation and the null test now.

      Created a file, ran a command 

      The mSPRT holds the Type I error at about 5% on 1,000 pure-null streams. In the main
      configuration (known σ², τ²=0.25) the false positive rate was 4.9% (±1.3%). Each stream
      had 2,000 observations per arm and was checked at every observation from n=30. Testing
      a fixed-sample z-test at every look on the same streams gave 47.8%.

      Configuration                                      mSPRT FPR                            Naive peeking FPR

      Known σ², τ²=0.25                                  4.9% (±1.3)                          47.8%

      Plug-in σ², τ²=0.25                                5.6% (±1.4)                          47.8%

      Known σ², τ²=0.01                                  2.6% (±1.0)                          47.8%

      Known σ², τ²=4.00                                  2.1% (±0.9)                          47.8%

      The ± values are 95% Monte Carlo intervals on the rate.
             Plug-in variance: 5.6% is within Monte Carlo noise of 5%. Estimating σ² from the
             data makes the guarantee asymptotic rather than exact, which is why min_n=30 is
             there.
             Rates below 5%: The Ville guarantee is an upper bound, not an exact rate. When the
             prior variance τ² is far from the data’s scale, the test rejects less often than it could. It
             also only watches 2,000 observations, while the bound covers an infinite horizon. A
             rate below 5% means lost power, not a bug.

      Why testing at every look inflates Type I error
      A fixed-sample test controls the false positive rate at one pre-chosen look. Under the null,
      the z-statistic Z_n behaves like a random walk scaled by 1/√n. It crosses ±1.96 with 5%
      probability at any single n. Across many looks the crossing events are positively
      correlated but not identical. Because the test stops at the first crossing, the error rate is
      the probability that the whole path ever crosses. By the law of the iterated logarithm, |Z_n|
      exceeds any fixed threshold infinitely often if you watch forever. The naive error
      therefore rises toward 1 as looks accumulate, and was 48% here.

      What the always-valid construction gives up
      Under H0 the likelihood ratio Λ_n is a nonnegative martingale with mean 1. Ville’s
                                  ∃
      inequality gives P( n: Λ_n ≥ 1/α) ≤ α, and this holds for any stopping rule. In exchange
      you give up:
        1. Power at any given n. You must beat a boundary of roughly √(n·log n), not a fixed
           1.96. For the same effect you need noticeably more samples than a fixed-horizon test
           planned in advance. The gain is that you can stop early on large effects and never
           have to pre-commit to n.
        2. Prior sensitivity. The mixing variance τ² decides where power is spent. A τ² near the
           true effect size is efficient, and a badly mismatched τ² is conservative, as the table
           shows.
        3. Exactness. The guarantee is α as an upper bound, so the test is conservative in
           practice.

        4. Interpretation. The p-value is min(1, 1/max Λ) and the CI is a time-uniform
           confidence sequence. Both are valid at any stopping time, but they are wider than
           fixed-sample versions and don’t mean “p at this n.”

      Not done yet
             Your 200-test portfolio and log structure. I found no files in the workspace, so I
             haven’t run on your portfolio. The function returns a columnar dict ( look, n,
             effect_hat, log_lambda, p_always_valid, ci_lo, ci_hi, reject ). Share your
             log schema and I’ll remap the column names. I can then report power and early-
             stopping savings against the known true effects.
             Prior choice. I set τ²=0.25 by hand. On the portfolio I’d pick it from the empirical
             spread of the true effects.

      The script is at /mnt/user-data/uploads/sequential.py . The function to call is
      msprt(x_control, x_treat, ...) , and null_fpr() is the pure-null test.

       Create a doc
     
```

### Code written by the model, v1 (`sequential.py`, unchanged)

```python
"""Always-valid sequential test: two-sample mixture SPRT (mSPRT), numpy/scipy only.

Model: control ~ (mu, sigma^2), treatment ~ (mu + theta, sigma^2).
H0: theta = 0. Mixing distribution on theta: N(0, tau2).
After n pairs, with D_n = mean_t - mean_c and V_n = 2*sigma^2/n:

    Lambda_n = sqrt(V_n / (V_n + tau2)) * exp( tau2 * D_n^2 / (2 * V_n * (V_n + tau2)) )

Under H0 Lambda_n is a nonnegative martingale with mean 1, so by Ville's
inequality P(exists n: Lambda_n >= 1/alpha) <= alpha for ANY stopping rule.
"""
import numpy as np
from scipy import stats


def msprt(x_control, x_treat, alpha=0.05, tau2=0.25, sigma2=None,
          min_n=30, look_every=1):
    """Run an mSPRT over a stream of paired observations.

    x_control, x_treat : 1-D arrays, arrival order (one pair per time step).
    tau2    : variance of the N(0, tau2) mixing prior on the effect.
    sigma2  : known per-arm variance. If None, the pooled running variance is
              plugged in (asymptotically valid; see min_n).
    min_n   : burn-in. Plug-in variance is unstable at tiny n, so no
              evidence is accumulated before min_n pairs per arm.
    look_every : evaluate the boundary every k pairs (1 = every observation).

    Returns a columnar log (dict of arrays, one entry per look):
      look, n, effect_hat, log_lambda, p_always_valid, ci_lo, ci_hi, reject
    plus scalars `stop_look` / `stop_n` (None if never rejected).
    The decision is monotone: once `reject` is True it stays True. The user
    may stop at first rejection; the log keeps going so the same function can
    be used to inspect the full path.
    """
    xc = np.asarray(x_control, float)
    xt = np.asarray(x_treat, float)
    if xc.shape != xt.shape or xc.ndim != 1:
        raise ValueError("streams must be 1-D arrays of equal length")
    N = xc.size
    n = np.arange(1, N + 1)

    mc, mt = np.cumsum(xc) / n, np.cumsum(xt) / n
    D = mt - mc

    if sigma2 is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            vc = (np.cumsum(xc**2) - n * mc**2) / (n - 1)
            vt = (np.cumsum(xt**2) - n * mt**2) / (n - 1)
        s2 = np.where(n >= 2, 0.5 * (vc + vt), np.nan)
    else:
        s2 = np.full(N, float(sigma2))

    V = 2.0 * s2 / n
    with np.errstate(divide="ignore", invalid="ignore"):
        loglam = 0.5 * np.log(V / (V + tau2)) + tau2 * D**2 / (2 * V * (V + tau2))
    loglam = np.where((n >= min_n) & np.isfinite(loglam), loglam, 0.0)

    run_max = np.maximum.accumulate(loglam)            # always-valid: running max
    p_av = np.minimum(1.0, np.exp(-run_max))           # p_n = min(1, min_k 1/Lambda_k)

    # Always-valid CI: invert the mSPRT (set of theta0 with Lambda(theta0) < 1/alpha),
    # computed in closed form for the normal mixture centered at theta0.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (V * (V + tau2) / tau2) * (2 * np.log(1 / alpha) + np.log((V + tau2) / V))
    half = np.sqrt(np.clip(r, 0, None))
    ci_lo, ci_hi = D - half, D + half

    idx = np.arange(look_every - 1, N, look_every)
    reject = run_max[idx] >= np.log(1 / alpha)
    log = {
        "look": np.arange(1, idx.size + 1),
        "n": n[idx],
        "effect_hat": D[idx],
        "log_lambda": loglam[idx],
        "p_always_valid": p_av[idx],
        "ci_lo": ci_lo[idx],
        "ci_hi": ci_hi[idx],
        "reject": reject,
    }
    hit = np.flatnonzero(reject)
    log["stop_look"] = int(hit[0]) + 1 if hit.size else None
    log["stop_n"] = int(log["n"][hit[0]]) if hit.size else None
    return log


def naive_peeking(x_control, x_treat, alpha=0.05, sigma2=1.0, min_n=30, look_every=1):
    """Baseline: fixed-sample z-test at every look, stop at first p < alpha."""
    xc, xt = np.asarray(x_control, float), np.asarray(x_treat, float)
    n = np.arange(1, xc.size + 1)
    z = (np.cumsum(xt) - np.cumsum(xc)) / n / np.sqrt(2 * sigma2 / n)
    idx = np.arange(look_every - 1, xc.size, look_every)
    ok = (n[idx] >= min_n) & (2 * stats.norm.sf(np.abs(z[idx])) < alpha)
    return bool(ok.any())


def null_fpr(n_streams=1000, N=2000, seed=0, **kw):
    """Empirical false-positive rate of msprt on pure-null N(0,1) streams."""
    rng = np.random.default_rng(seed)
    rej = np.empty(n_streams, bool)
    naive = np.empty(n_streams, bool)
    for i in range(n_streams):
        xc, xt = rng.standard_normal(N), rng.standard_normal(N)   # theta = 0
        rej[i] = msprt(xc, xt, **kw)["stop_look"] is not None
        naive[i] = naive_peeking(xc, xt, alpha=kw.get("alpha", 0.05),
                                 sigma2=kw.get("sigma2") or 1.0,
                                 min_n=kw.get("min_n", 30),
                                 look_every=kw.get("look_every", 1))
    return rej.mean(), naive.mean()


if __name__ == "__main__":
    for label, kw in [("known sigma^2, tau2=0.25", dict(sigma2=1.0, tau2=0.25)),
                      ("plug-in sigma^2, tau2=0.25", dict(sigma2=None, tau2=0.25)),
                      ("known sigma^2, tau2=0.01", dict(sigma2=1.0, tau2=0.01)),
                      ("known sigma^2, tau2=4.00", dict(sigma2=1.0, tau2=4.0))]:
        f, nv = null_fpr(1000, 2000, seed=1, **kw)
        se = np.sqrt(f * (1 - f) / 1000)
        print(f"{label:28s} mSPRT FPR={f:.3f} (±{1.96*se:.3f})   naive peeking FPR={nv:.3f}")
```

### What I rejected / found wrong in v1
- It never saw my log (it said no files were found), so it invented its own per-look format and did not run on my 200 tests.
- It set the mixing variance tau^2 = 0.25 by hand, a typical effect of 0.5 sigma, when my real effects average about 0.10 sigma.
- It simulated every stream at a fixed 2,000 observations per arm, not my 50 to 2,000 range.
- Its "±1.3%" is a 95% interval (1.96 x SE), not the Monte Carlo SE.

---

## Version 2

### My revision and why
v1 left my log structure and effect scale unsaid, so the model guessed. In v2 I kept the given prompt word for word and added, at the end of the [Iterate] section, my log's columns, my effect scale (sigma = 1, n_per_arm 50 to 2,000, real effects at least 0.03 with mean about 0.10), a request to choose tau^2 from that scale and say why, and to build each stream from a log row's n_per_arm. The output changed: tau^2 came from the log (0.0148 instead of 0.25), streams used my sample sizes, and the null false positive rate went from 4.9% to 0.6%.

### Prompt v2 (as I sent it, in a new chat)

```text
[Prep] Act as an experimentation platform engineer who has implemented sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true effects. Implement an always-valid sequential test -- a mixture sequential probability ratio test or an alpha-spending boundary -- that allows continuous monitoring without inflating the Type I error rate. Give me a function that takes a stream of observations and returns the stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function must be testable against a pure-null stream. My log has one row per experiment with columns is_real, true_effect, n_per_arm, se, estimate, z, p_value, power, essentially_zero. Effects are in units of sigma = 1, n_per_arm ranges from 50 to 2,000, and real effects are at least 0.03 with a mean of about 0.10. Choose the mixing variance tau^2 from that scale instead of a hand-picked value and say why, and build each stream from a log row's n_per_arm (a pure-null stream uses true_effect = 0).

[Mechanism Check] Explain why the naive approach of testing at every look inflates Type I error, and precisely what the always-valid construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical false positive rate. If it is not close to 5%, tell me what is wrong.
```

### Raw exchange, v2 (opener, reply, and the model's final answer)

```text
Prompt 2         

                                                           Hello Claude, prepare for a prompt I'm about to send you a
                                                           prompt for my causal machine learning course.

   Ready when you are. Send over the prompt for your causal ML course, along with any files,
   datasets, or notebooks it relies on, and I’ll get started.

[prompt v2 sent here, as above]

Looking for the attached log file.
      I’ll check for your log file first, then build the mSPRT, run the null simulation, and report the
      false positive rate.

      Looked for files, updated tasks 
      No log file is attached here, so I’ll build a synthetic 200-row log with your column structure
      and swap yours in via a loader. Writing the implementation now.

      Created a file, shared a file, and 3 more steps 
      The empirical false positive rate on 1,000 pure-null streams is 0.6% (95% CI 0.3–1.3%), not
      5%. The test is valid but conservative, and the shortfall is a property of finite horizons, not a
      bug.
      No log file was attached, so everything ran on a synthetic 200-row log with your column
      structure (n_per_arm 50–2,000, real effects ≥ 0.03 with mean ≈ 0.097). Run python

      msprt.py your_log.csv and it uses your data instead. The code is in msprt.py .

      What I built
              msprt(treatment, control, tau2, alpha, looks=None) takes a stream and returns
             the stopping decision at each look: reject , log_lambda , always_valid_p , estimate ,
             and stop_at . Any look schedule is valid because the statistic is computed on every
             observation.
              OnlineMSPRT is an O(1)-state version for production, one update(x_t, y_t) per pair.
             It is asserted to match the vectorised version.
             Streams are built from each row’s n_per_arm , with true_effect = 0 for a pure-null
             stream.

      Choosing tau²
      I set tau² = E[true_effect² | real] = 0.0148 (τ = 0.122), computed from the log. The mixing prior
      is centered at zero, so the right matching quantity is the second moment of real effects
      (RMS), not the mean squared. A too-small tau² wastes power on effects above the prior’s
      scale. A too-large tau² pays an Occam penalty, (1 + τ²I)^(-1/2), at every look.
      This choice is not power-optimal on this log. A 10× larger tau² gave 17.9% power against
      15.5% for mine. The log’s median n is small, so a wider prior helps. The null FPR stayed at or
      below 2.6% for every tau² tried, because validity doesn’t depend on tau² at all.

      Why the FPR is 0.6% and not 5%
      The Type I guarantee is P(sup_n Λ_n ≥ 1/α) ≤ α over an infinite horizon, and it is an inequality,
      not an equality. Your streams stop at n ≤ 2,000, so most of that probability mass lies beyond
      the window. The same code gives:

      Horizon (n per arm)                                 mSPRT FPR               Naive peek-every-look

      50                                                  0.0%                    31%

      200                                                 0.3%                    43%

      2,000                                               2.4%                    56%

      20,000                                              4.2%                    n/a

      200,000                                             4.9%                    n/a

      The FPR climbs toward 5% as the horizon grows, so the construction is correctly calibrated.
      As a check on the simulator, the fixed-horizon z-test at the end of each stream gave 5.8%,
      about one Monte Carlo SE above 5%.

      Why the naive approach inflates Type I error
      A fixed-horizon test controls error at one pre-specified n. If you test at every look, you get
      many chances to cross ±1.96 on a random walk, and the sup of |Z_n| over n grows without
      bound (law of the iterated logarithm). On your streams this gave 46% false positives. The
      looks are correlated, so the inflation is less than the number of looks, but it is far above 5%.

      What the always-valid construction gives up
             Power. On the log’s real experiments the mSPRT rejected 15.5% of the time, against
             31.6% for the fixed-horizon test at the full n. The Occam penalty and the boundary cost
             of validity at every n are the price. The test stops early when it does reject, at a median of
             48% of the planned n.
             Calibration. At finite horizons the test is conservative, as shown above. The validity
             guarantee is not exact at any specific n.
             Prior dependence. Power depends on how well tau² matches the true effect scale.
             Validity does not.
             Estimates. The running estimate is biased at a data-dependent stopping time. The
              always_valid_p is valid at any stop, but a naive confidence interval at the stopping
             point is not.

      Next I’d swap in your real log and re-check the power table, since the 15.5% vs 31.6% gap is
      the number your course will care about most.

          Msprt    PY

     
```

### Code written by the model, v2 (`msprt.py`, unchanged)

```python
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
```

---

## Changes I made to the model's output
1. Saved the model's v2 code to `src/msprt.py` with a `%%writefile` line at the top of the notebook cell. The code itself is unchanged.
2. Wrote a short check cell in the notebook that turns my log (a DataFrame) into the dict of arrays the model's functions expect, calls its `choose_tau2` and `evaluate` on 1,000 pure-null streams with my own seed, and computes the Monte Carlo SE as sqrt(p(1-p)/1000), since the model reported a Wilson interval instead.
3. The v1 code was not run or changed; it is kept here only as the record.

## What I found wrong or missing in v2
- It computes tau^2 from `true_effect`, which a real client never observes; this works in a simulation, but a real team would have to guess the effect scale.
- It ran on a synthetic 200-row log (about 50% real ideas), not my portfolio (20% real), so its 0.6% is not my number; I reran the check on my own log.
- It added things I did not ask for (an online version, a CSV loader, horizon and tau^2 sweeps).
