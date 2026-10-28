"""Experiment-portfolio simulator and decision rules (ECON 5200 PS2)."""
import numpy as np
import pandas as pd
from scipy import stats

ZERO_BAND = 0.03   # |true effect| below this counts as "essentially zero"


def simulate_portfolio(n: int = 200, seed: int = 42, p_dud: float = 0.80, dud_sd: float = 0.01,
                       real_min: float = 0.03, real_scale: float = 0.07, n_min: int = 50,
                       n_max: int = 2_000, sigma: float = 1.0, alpha: float = 0.05) -> pd.DataFrame:
    """Simulate n experiments and return a log with the TRUE effect attached."""
    gen = np.random.default_rng(seed)
    is_real = gen.random(n) >= p_dud
    true_effect = np.where(is_real, real_min + gen.exponential(real_scale, n), gen.normal(0.0, dud_sd, n))
    n_per_arm = np.exp(gen.uniform(np.log(n_min), np.log(n_max), n)).round().astype(int)
    se = sigma * np.sqrt(2.0 / n_per_arm)
    estimate = true_effect + gen.normal(0.0, se)
    z = estimate / se
    power = stats.norm.sf(stats.norm.isf(alpha / 2) - (real_min + real_scale) / se)
    return pd.DataFrame({
        "is_real": is_real, "true_effect": true_effect, "n_per_arm": n_per_arm, "se": se,
        "estimate": estimate, "z": z, "p_value": 2 * stats.norm.sf(np.abs(z)), "power": power,
        "essentially_zero": np.abs(true_effect) < ZERO_BAND,
    })


def apply_rule(log: pd.DataFrame, rule: dict) -> pd.Series:
    """Return True for each experiment the rule ships. {"alpha": 0.05} is the threshold rule: ship if p < alpha and the estimate is positive.
    Adding "min_power": 0.3 also requires power >= 0.3 (the minimum power-floor rule applied)."""

    ship = (log["p_value"] < rule["alpha"]) & (log["estimate"] > 0)
    if "min_power" in rule:
        ship = ship & (log["power"] >= rule["min_power"])
    return ship


def evaluate(log: pd.DataFrame, rule: dict) -> dict:
    """Return the realised return (sum of true effects shipped), number shipped and false-ship rate."""
    won = log[apply_rule(log, rule)]
    return {"realised_return": float(won["true_effect"].sum()),
            "n_shipped": len(won),
            "false_ship_rate": float(won["essentially_zero"].mean()) if len(won) else 0.0}


if __name__ == "__main__":
    log = simulate_portfolio(200, 42)
    client, floor = {"alpha": 0.05}, {"alpha": 0.05, "min_power": 0.3}
    print("client     ", evaluate(log, client))
    print("power floor", evaluate(log, floor))

    assert "true_effect" in log.columns and len(log) == 200               # truth is attached
    assert apply_rule(log, floor).sum() <= apply_rule(log, client).sum()  # a power floor never ships more
    assert simulate_portfolio(200, 42).equals(log)                        # same seed, same portfolio
    print("all assertions passed")
