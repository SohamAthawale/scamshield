# xgboost MUST be imported before anything imports torch. On macOS, importing torch first and then
# training an XGBoost model in the same process segfaults (exit 139) because two OpenMP runtimes get
# loaded. Importing xgboost first avoids it. Local-only issue; the target (Linux s390x) is unaffected.
import xgboost  # noqa: F401  isort: skip

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scamshield.data import FORMATS

REAL_AML_DIR = Path(__file__).resolve().parents[1] / "shared" / "data" / "aml"

# ------------------------------------------------------------------------------- synthetic AMLworld data
# 8 days: train < 09-05 <= val < 09-07 <= test < 09-09 (= END). Rows on 09-09/09-10 exist only to test the cut.
START = pd.Timestamp("2022-09-01")
SPLITS = ("2022-09-05", "2022-09-07")
END = "2022-09-09"
HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,"
          "Payment Currency,Payment Format,Is Laundering")
BANKS = ["010", "03208", "001", "0119", "021174"]
# USD per unit. Yen never converts straight to USD, so FeatureContext.fit must walk Yen -> Euro -> USD.
USD_PER = {"US Dollar": 1.0, "Euro": 1.1, "Yen": 0.0075}
# the two demo customers: >= 6 ordinary payments each before the live days, then one big first payment
# to a brand-new payee during the live days (fires the big_new_payee rule)
CUSTOMER_1 = ("0119", "8C0FFEE01")
CUSTOMER_2 = ("001", "8C0FFEE02")
LIVE_MULE = ("021174", "80BADBAD4")


def _acct(rng, used: set) -> str:
    while True:
        a = f"8{rng.integers(0, 16 ** 8):08X}"
        if a not in used:
            used.add(a)
            return a


@dataclass
class Row:
    ts: pd.Timestamp
    from_bank: str
    from_acct: str
    to_bank: str
    to_acct: str
    amount_paid: float
    currency_paid: str
    currency_received: str
    payment_format: str
    label: int = 0

    @property
    def amount_received(self) -> float:
        return round(self.amount_paid * USD_PER[self.currency_paid] / USD_PER[self.currency_received], 2)

    def csv(self) -> str:
        return ",".join([self.ts.strftime("%Y/%m/%d %H:%M"), self.from_bank, self.from_acct, self.to_bank,
                         self.to_acct, f"{self.amount_received:.2f}", self.currency_received,
                         f"{self.amount_paid:.2f}", self.currency_paid, self.payment_format, str(self.label)])


def make_aml_rows(seed: int = 0, n_accounts: int = 30) -> tuple[list[Row], list[tuple[str, list[Row]]]]:
    """Normal traffic + laundering attempts (FAN-IN in each period, one CYCLE). Returns (rows, attempts)."""
    rng = np.random.default_rng(seed)
    used = {CUSTOMER_1[1], CUSTOMER_2[1], LIVE_MULE[1]}
    accts = [(str(rng.choice(BANKS)), _acct(rng, used)) for _ in range(n_accounts)]
    accts += [CUSTOMER_1, CUSTOMER_2]
    cur = {a: str(rng.choice(["US Dollar"] * 7 + ["Euro"] * 2 + ["Yen"])) for a in accts}
    cur[CUSTOMER_1] = cur[CUSTOMER_2] = "US Dollar"
    usual = {a: float(np.exp(rng.uniform(3, 8))) for a in accts}
    usual[CUSTOMER_1] = usual[CUSTOMER_2] = 100.0
    payees = {a: [accts[j] for j in rng.choice(len(accts), 3, replace=False) if accts[j] != a][:2] for a in accts}
    rows: list[Row] = []

    def pay(ts, s, r, amount, fmt, label=0, paid=None):
        paid = paid or cur.get(s, "US Dollar")
        recv = cur.get(r, paid)
        if {paid, recv} == {"Yen", "US Dollar"}:
            recv = paid                                        # no direct Yen<->USD conversions in the data
        row = Row(ts, s[0], s[1], r[0], r[1], round(float(amount), 2), paid, recv, fmt, label)
        rows.append(row)
        return row

    fmts = ["ACH", "Cheque", "Credit Card", "Wire", "Cash", "Bitcoin"]
    p = np.array([.35, .2, .2, .1, .1, .05])
    for day in range(10):                                     # includes 09-09 and 09-10 (cut by END)
        for a in accts:
            for _ in range(int(rng.integers(1, 3))):
                ts = START + pd.Timedelta(days=day, minutes=int(rng.integers(0, 24 * 60)))
                if rng.random() < 0.06:                        # reinvestment = transfer to own account
                    pay(ts, a, a, usual[a] * rng.lognormal(0, .3), "Reinvestment")
                else:
                    pay(ts, a, payees[a][int(rng.integers(len(payees[a])))], usual[a] * rng.lognormal(0, .3),
                        str(rng.choice(fmts, p=p)))

    attempts = []

    def fan_in(day: int, mule, n_senders: int):
        att = []
        for k in range(n_senders):
            s = (str(rng.choice(BANKS)), _acct(rng, used))
            att.append(pay(START + pd.Timedelta(days=day, hours=1 + k, minutes=7), s, mule,
                           rng.uniform(4000, 9000), "ACH", 1, paid="US Dollar"))
        onward = (str(rng.choice(BANKS)), _acct(rng, used))
        att.append(pay(START + pd.Timedelta(days=day, hours=n_senders + 2), mule, onward,
                       sum(r.amount_paid for r in att) * 0.97, "Wire", 1, paid="US Dollar"))
        attempts.append(("FAN-IN", att))

    fan_in(2, ("010", _acct(rng, used)), 6)                   # train
    fan_in(5, ("001", _acct(rng, used)), 6)                   # val
    fan_in(7, LIVE_MULE, 5)                                   # test (live): LIVE_MULE is a known mule
    ring = [(str(rng.choice(BANKS)), _acct(rng, used)) for _ in range(3)]
    att = [pay(START + pd.Timedelta(days=3, hours=h), ring[h], ring[(h + 1) % 3], 15000 - 100 * h, "Wire", 1,
               paid="US Dollar") for h in range(3)]
    attempts.append(("CYCLE", att))

    # the customers: ordinary payments, then a big first payment to a new payee in the live days
    for cust, day in ((CUSTOMER_1, 6), (CUSTOMER_2, 7)):   # 09-07 and 09-08: live days
        for d in range(7):
            pay(START + pd.Timedelta(days=d, hours=9), cust, payees[cust][0], 100.0 + d, "ACH")
        pay(START + pd.Timedelta(days=day, hours=10), cust, ("03208", _acct(rng, used)), 5000.0, "Wire")
    # two payments in the same minute (stable order must be kept), written in this order
    same = START + pd.Timedelta(days=1, hours=12, minutes=34)
    pay(same, accts[0], accts[1], 11.11, "Cheque")
    pay(same, accts[2], accts[3], 22.22, "Cheque")
    return rows, attempts


def write_aml(raw_dir: Path, seed: int = 0) -> tuple[list[Row], list]:
    rows, attempts = make_aml_rows(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(rows)), key=lambda i: rows[i].ts)   # stable: same-minute rows keep insertion order
    lines = [rows[i].csv() for i in order]
    lines[10], lines[11] = lines[11], lines[10]                  # a little disorder: the loader must sort
    (raw_dir / "XS-Small_Trans.csv").write_text(HEADER + "\n" + "\n".join(lines) + "\n")
    pat = []
    for kind, att in attempts:
        pat.append(f"BEGIN LAUNDERING ATTEMPT - {kind}:  Max {len(att)}-degree {kind.title()}")
        pat += [r.csv() for r in att]
        pat.append(f"END LAUNDERING ATTEMPT - {kind}")
        pat.append("")
    (raw_dir / "XS-Small_Patterns.txt").write_text("\n".join(pat))
    return rows, attempts


@dataclass
class AML:
    raw_dir: Path
    rows: list
    attempts: list
    tx: pd.DataFrame
    split: np.ndarray
    ctx: object
    X: np.ndarray
    y: np.ndarray
    models_dir: Path


def build_tiny_models(models_dir: Path, tx: pd.DataFrame, split: np.ndarray, X: np.ndarray, ctx,
                      seed: int = 0) -> dict:
    """Random autoencoder over ANOMALY_FEATURES + 8-round XGBoost over FEATURES + thresholds."""
    import torch
    import xgboost as xgb

    from scamshield.features import ANOMALY_FEATURES, FEATURES
    from scamshield.inference.anomaly import NumpyAnomalyScorer
    from scamshield.models import Artifacts, Standardizer, build_torch_autoencoder, torch_to_numpy_weights

    models_dir.mkdir(parents=True, exist_ok=True)
    art = Artifacts(models_dir)
    ae_idx = [FEATURES.index(f) for f in ANOMALY_FEATURES]
    Standardizer.fit(ANOMALY_FEATURES, X[:, ae_idx]).save(art.scaler)
    torch.manual_seed(seed)
    model = build_torch_autoencoder(len(ANOMALY_FEATURES))
    torch.save(model.state_dict(), art.ae_torch)
    np.savez(art.ae_numpy, **torch_to_numpy_weights(model))

    y = tx["label"].values.astype(int)
    d = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    booster = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.5, "nthread": 1, "seed": seed},
                        d, num_boost_round=8)
    booster.save_model(str(art.risk_model))
    art.risk_meta.write_text(json.dumps({"features": FEATURES, "trees": 8}))
    art.feature_context.write_text(ctx.to_json())

    test = split == "test"
    risk = booster.inplace_predict(X).astype(np.float64)
    anomaly, _, _ = NumpyAnomalyScorer(art).score(X[:, ae_idx])
    uniq = np.unique(risk[test])
    thresholds = {"anomaly_threshold": float(np.quantile(anomaly[test], 0.9)),
                  "warn_threshold": float(uniq[-4]),           # the top four distinct test risks warn ...
                  "hold_threshold": float(uniq[-2])}           # ... and the top two are held
    art.thresholds.write_text(json.dumps(thresholds))
    return thresholds


@pytest.fixture(scope="session")
def aml(tmp_path_factory) -> AML:
    from scamshield import data as D
    from scamshield.features import FeatureContext, FeatureEngine

    root = tmp_path_factory.mktemp("aml")
    rows, attempts = write_aml(root / "raw")
    tx = D.load_transactions(str(root / "raw"), end=END)
    split = D.time_split(tx["ts"], SPLITS)
    ctx = FeatureContext.fit(tx[split == "train"])
    X = FeatureEngine(ctx).run(tx)
    build_tiny_models(root / "models", tx, split, X, ctx)
    return AML(root / "raw", rows, attempts, tx, split, ctx, X, tx["label"].values.astype(int), root / "models")


@pytest.fixture
def fresh_models_dir(tmp_path, aml) -> Path:
    dst = tmp_path / "models"
    shutil.copytree(aml.models_dir, dst)
    return dst


def scorers(models_dir: Path, backend: str = "numpy"):
    from scamshield.inference import load_anomaly_scorer, load_risk_scorer

    return load_anomaly_scorer(models_dir, backend), load_risk_scorer(models_dir)


def policy_for(models_dir: Path):
    from scamshield.fusion import Policy

    t = json.loads((models_dir / "thresholds.json").read_text())
    return Policy(t["anomaly_threshold"], t["warn_threshold"], t["hold_threshold"])


# ------------------------------------------------------------------------------- API environment
DEMO_USERS = {
    "analyst": {"role": "analyst"},
    "auditor": {"role": "auditor"},
    "customer": {"role": "customer", "account": ":".join(CUSTOMER_1)},
    "customer2": {"role": "customer", "account": ":".join(CUSTOMER_2)},
}


class ApiEnv:
    """Everything create_app needs under one tmp dir: models, history.csv, stream file, secrets, policy."""

    def __init__(self, root: Path, aml: AML):
        from scamshield.security import generate_secrets, read_secret
        from scamshield.training import write_history, write_stream

        self.root, self.aml = root, aml
        self.models = root / "models"
        shutil.copytree(aml.models_dir, self.models)
        self.data_in = root / "data-in"
        self.data_in.mkdir()
        self.history = write_history(aml.tx, aml.split, self.data_in)
        self.stream_file = write_stream(aml.tx, aml.split, self.data_in)
        self.secrets = root / "secrets"
        self.passwords = generate_secrets(self.secrets, DEMO_USERS)
        self.service_token = read_secret("service_token", self.secrets)
        self.policy = root / "policy.json"
        self.policy.write_text(json.dumps({"_comment": "null = learned", "anomaly_threshold": None,
                                           "warn_threshold": None, "hold_threshold": None}))
        self.db = root / "data" / "scamshield.db"
        self.checkpoints = root / "data" / "audit_checkpoints.log"

    def settings(self, **over):
        from scamshield.api.settings import Settings

        kw = dict(models_dir=self.models, policy_file=self.policy, history_file=self.history,
                  history_end=SPLITS[1], db_path=self.db, checkpoint_path=self.checkpoints,
                  secrets_dir=self.secrets, anomaly_backend="numpy", nnpa_min_batch=256, batch_window_ms=2.0,
                  max_batch=512, jwt_ttl_minutes=60)
        kw.update(over)
        return Settings(**kw)

    def client(self, **over):
        from fastapi.testclient import TestClient

        from scamshield.api.main import create_app

        return TestClient(create_app(self.settings(**over)))

    def stream(self) -> list[dict]:
        from scamshield.simulator import load_stream

        return load_stream(self.stream_file, None)

    def pseudo(self):
        from scamshield.security import Pseudonymizer, read_secret

        return Pseudonymizer(bytes.fromhex(read_secret("pseudo_key", self.secrets)))

    def envelope(self):
        from scamshield.security import load_keyring

        return load_keyring(self.secrets)


@pytest.fixture
def api_env(tmp_path, aml) -> ApiEnv:
    return ApiEnv(tmp_path / "api", aml)


def login(client, user: str, env: ApiEnv) -> dict:
    r = client.post("/auth/login", json={"username": user, "password": env.passwords[user]})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def bearer(headers: dict) -> str:
    return headers["Authorization"].split()[1]


def service_headers(env: ApiEnv) -> dict:
    return {"Authorization": f"Bearer {env.service_token}"}


def raw_account_numbers(aml: AML) -> set[str]:
    """Every raw account number in the synthetic data (9 hex digits starting with 8)."""
    return set(aml.tx["from_acct"]) | set(aml.tx["to_acct"])
