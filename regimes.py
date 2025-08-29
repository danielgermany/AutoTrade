from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type, Mapping
import numpy as np
import pandas as pd
import sys, os

# Ensure local import works if placed alongside signal_generator.py
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

# Import indicators from your existing module
from signal_generator import (
    dmi_adx, atr, bollinger, keltner, ichimoku, ema, supertrend, macd, _to_np
)

# ----------------------------
# Column inference (like the one in signal_generator, kept local to avoid cycles)
# ----------------------------
def _infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None
    return {
        'open': find('open', 'o'),
        'high': find('high', 'h'),
        'low':  find('low', 'l'),
        'close':find('close', 'c'),
        'volume': find('volume', 'vol', 'v'),
        'datetime': find('datetime', 'timestamp', 'time', 'date')
    }

# ----------------------------
# Utilities
# ----------------------------
def _nan_to_num(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    y = x.astype(float).copy()
    y[np.isnan(y)] = fill
    return y

def _pct_change(x: np.ndarray, lag=1) -> np.ndarray:
    prev = np.roll(x, lag).astype(float)
    prev[:lag] = np.nan
    return (x - prev) / np.where(np.abs(prev) < 1e-12, np.nan, prev)

def _rolling_slope(x: np.ndarray, n: int) -> np.ndarray:
    """Simple slope via (x - x.shift(n))/n; robust enough for regime filters."""
    prev = np.roll(x, n).astype(float)
    prev[:n] = np.nan
    return (x - prev) / float(n)

def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    rng = hi - lo if (hi - lo) > 1e-12 else 1.0
    z = (x - lo) / rng
    z[z < 0] = 0; z[z > 1] = 1
    return z

def _series_like(ref: pd.Series, arr: np.ndarray, name: str) -> pd.Series:
    return pd.Series(arr, index=ref.index, name=name)

# ----------------------------
# Data carriers
# ----------------------------
@dataclass
class RegimeConfig:
    adx_trend_gate: float = 25.0
    adx_range_gate: float = 18.0
    ema_len_for_slope: int = 20
    atr_len: int = 14
    bb_len: int = 20
    bb_mult: float = 2.0
    kc_len: int = 20
    kc_mult: float = 2.0
    squeeze_min_bars: int = 3
    breakout_atr_pct_gate: float = 2.0    # e.g., 2% of price
    confirm_bars: int = 2                 # bars to confirm entering a primary regime
    exit_bars: int = 2                    # bars to confirm exiting a regime (hysteresis)

@dataclass
class RegimeResult:
    """Returned by the engine."""
    regimes_df: pd.DataFrame   # columns: primary, direction, confidence, tags (list), plus helpful fields
    features_df: pd.DataFrame  # computed features used for detection

# ----------------------------
# Registries and base classes
# ----------------------------
PRIMARY_REGISTRY: Dict[str, "BasePrimaryDetector"] = {}
TAG_REGISTRY: Dict[str, "BaseTagDetector"] = {}

def register_primary(cls: Type["BasePrimaryDetector"]):
    PRIMARY_REGISTRY[cls.key] = cls
    return cls

def register_tag(cls: Type["BaseTagDetector"]):
    TAG_REGISTRY[cls.key] = cls
    return cls

class BasePrimaryDetector:
    """Primary regimes are mutually exclusive; exactly one wins each bar by confidence & priority.

    Implementations must set:
      - key: str (unique)
      - priority: int (higher gets preference on ties)
    """
    key: str = "base"
    priority: int = 0

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Return (is_active: bool Series, direction: {-1,0,1} Series, confidence: [0..1] Series).
        Length must equal df length.
        """
        raise NotImplementedError

class BaseTagDetector:
    """Tags are orthogonal overlays; many can be true at once.
    Implementations must set key: str.
    """
    key: str = "tag_base"

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
        """Return bool Series for whether the tag applies on each bar."""
        raise NotImplementedError

# ----------------------------
# Feature computation (once)
# ----------------------------
def compute_features(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    colmap = _infer_columns(df)
    for r in ['open','high','low','close']:
        if colmap[r] is None:
            raise ValueError(f"Missing required column for '{r}'. Available: {list(df.columns)}")

    o = df[colmap['open']]; h = df[colmap['high']]; l = df[colmap['low']]; c = df[colmap['close']]
    v = df[colmap['volume']] if colmap['volume'] is not None else pd.Series(np.nan, index=df.index)

    # ADX
    _, _, adx = dmi_adx(h, l, c, length=int(cfg.atr_len))
    # ATR%
    atr_vals = atr(h, l, c, length=int(cfg.atr_len))
    atr_pct = 100.0 * _to_np(atr_vals).astype(float) / np.clip(_to_np(c).astype(float), 1e-12, np.inf)
    # EMA and slope
    ema20 = ema(c, length=int(cfg.ema_len_for_slope))
    ema_slope = _rolling_slope(_to_np(ema20).astype(float), n=int(cfg.ema_len_for_slope))
    # Bollinger & width
    basis, bb_up, bb_lo, bb_width = bollinger(c, length=int(cfg.bb_len), mult=float(cfg.bb_mult))
    # Keltner
    kc_mid, kc_up, kc_lo = keltner(h, l, c, length=int(cfg.kc_len), mult=float(cfg.kc_mult))
    # Squeeze flag
    squeeze_on = (_to_np(bb_up) < _to_np(kc_up)) & (_to_np(bb_lo) > _to_np(kc_lo))
    # Supertrend + direction
    st_line, st_dir = supertrend(h, l, c, length=10, multiplier=3.0)
    # MACD basic
    macd_line, macd_sig, macd_hist = macd(c, fast=12, slow=26, signal=9)
    # Ichimoku cloud
    ich = ichimoku(h, l, c, tenkan=9, kijun=26, senkou_b=52)
    span_a, span_b = ich['span_a'], ich['span_b']
    above_cloud = (_to_np(c) > np.maximum(_to_np(span_a), _to_np(span_b)))
    below_cloud = (_to_np(c) < np.minimum(_to_np(span_a), _to_np(span_b)))

    feats = pd.DataFrame({
        'adx': _to_np(adx),
        'atr_pct': atr_pct,
        'ema20': _to_np(ema20),
        'ema20_slope': ema_slope,
        'bb_up': _to_np(bb_up),
        'bb_lo': _to_np(bb_lo),
        'bb_width': _to_np(bb_width),
        'kc_mid': _to_np(kc_mid),
        'kc_up': _to_np(kc_up),
        'kc_lo': _to_np(kc_lo),
        'squeeze_on': squeeze_on.astype(float),
        'st_dir': _to_np(st_dir),
        'st_line': _to_np(st_line),
        'macd_line': _to_np(macd_line),
        'macd_sig': _to_np(macd_sig),
        'macd_hist': _to_np(macd_hist),
        'span_a': _to_np(span_a),
        'span_b': _to_np(span_b),
        'above_cloud': above_cloud.astype(float),
        'below_cloud': below_cloud.astype(float),
        'close': _to_np(c),
    }, index=df.index)
    return feats

# ----------------------------
# Primary detectors (reference implementations)
# ----------------------------
@register_primary
class TrendUp(BasePrimaryDetector):
    key = "trend_up"
    priority = 80

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig):
        adx = feats['adx']
        # bullish confirmations
        bullish = (feats['st_dir'] > 0) | (feats['macd_line'] > feats['macd_sig']) | (feats['above_cloud'] > 0)
        active = (adx >= cfg.adx_trend_gate) & bullish
        # confidence rises with ADX above gate
        conf = _normalize01(_to_np(adx) - cfg.adx_trend_gate)
        conf = np.where(active, conf, 0.0)
        direction = np.where(active, 1.0, 0.0)
        return _series_like(adx, active.astype(bool), 'active'), _series_like(adx, direction, 'dir'), _series_like(adx, conf, 'conf')

@register_primary
class TrendDown(BasePrimaryDetector):
    key = "trend_down"
    priority = 80

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig):
        adx = feats['adx']
        bearish = (feats['st_dir'] < 0) | (feats['macd_line'] < feats['macd_sig']) | (feats['below_cloud'] > 0)
        active = (adx >= cfg.adx_trend_gate) & bearish
        conf = _normalize01(_to_np(adx) - cfg.adx_trend_gate)
        conf = np.where(active, conf, 0.0)
        direction = np.where(active, -1.0, 0.0)
        return _series_like(adx, active.astype(bool), 'active'), _series_like(adx, direction, 'dir'), _series_like(adx, conf, 'conf')

@register_primary
class Breakout(BasePrimaryDetector):
    key = "breakout"
    priority = 70

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig):
        c = feats['close'].values
        atr_pct = feats['atr_pct'].values
        up = feats['kc_up'].values
        lo = feats['kc_lo'].values
        bull = (c > up) & (atr_pct >= cfg.breakout_atr_pct_gate)
        bear = (c < lo) & (atr_pct >= cfg.breakout_atr_pct_gate)
        active = bull | bear
        direction = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
        # confidence from ATR% magnitude beyond the gate
        conf = _normalize01(atr_pct - cfg.breakout_atr_pct_gate)
        conf = np.where(active, conf, 0.0)
        return _series_like(feats['close'], active.astype(bool), 'active'), _series_like(feats['close'], direction, 'dir'), _series_like(feats['close'], conf, 'conf')

@register_primary
class Squeeze(BasePrimaryDetector):
    key = "squeeze"
    priority = 60

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig):
        sq = feats['squeeze_on'].values > 0
        # enforce min bars in squeeze
        sq_run = np.zeros_like(sq, dtype=bool)
        cnt = 0
        for i, s in enumerate(sq):
            cnt = cnt + 1 if s else 0
            sq_run[i] = (cnt >= cfg.squeeze_min_bars)
        active = sq_run
        direction = np.zeros_like(sq, dtype=float)  # unknown
        # confidence increases with how narrow the BB width is vs history
        conf = 1.0 - _normalize01(feats['bb_width'].values)
        conf = np.where(active, conf, 0.0)
        return _series_like(feats['close'], active.astype(bool), 'active'), _series_like(feats['close'], direction, 'dir'), _series_like(feats['close'], conf, 'conf')

@register_primary
class Range(BasePrimaryDetector):
    key = "range"
    priority = 50

    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig):
        adx = feats['adx'].values
        slope = np.abs(feats['ema20_slope'].values)
        low_adx = adx <= cfg.adx_range_gate
        flat = slope <= np.nanpercentile(slope, 60)  # loose flatness heuristic
        active = low_adx & flat
        direction = np.zeros_like(adx, dtype=float)
        # confidence higher when ADX is well below gate and slope is flatter
        conf = (1.0 - _normalize01(adx)) * (1.0 - _normalize01(slope))
        conf = np.where(active, conf, 0.0)
        return _series_like(feats['close'], active.astype(bool), 'active'), _series_like(feats['close'], direction, 'dir'), _series_like(feats['close'], conf, 'conf')

# ----------------------------
# Tag detectors (examples)
# ----------------------------
@register_tag
class PullbackUp(BaseTagDetector):
    key = "pullback_up"
    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
        # Uptrend pullback: supertrend dir > 0 and RSI-like condition via MACD weakness
        cond = (feats['st_dir'] > 0) & (feats['macd_line'] < feats['macd_sig'])
        return cond.astype(bool)

@register_tag
class PullbackDown(BaseTagDetector):
    key = "pullback_down"
    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
        cond = (feats['st_dir'] < 0) & (feats['macd_line'] > feats['macd_sig'])
        return cond.astype(bool)

@register_tag
class Accumulation(BaseTagDetector):
    key = "accumulation"
    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
        # proxy: price flat-ish but MACD histogram rising
        flat = np.abs(feats['ema20_slope'].values) <= np.nanpercentile(np.abs(feats['ema20_slope'].values), 50)
        rising = feats['macd_hist'].values > 0
        return (flat & rising)

@register_tag
class Distribution(BaseTagDetector):
    key = "distribution"
    def detect(self, df: pd.DataFrame, feats: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
        flat = np.abs(feats['ema20_slope'].values) <= np.nanpercentile(np.abs(feats['ema20_slope'].values), 50)
        falling = feats['macd_hist'].values < 0
        return (flat & falling)

# ----------------------------
# Engine
# ----------------------------
class RegimeEngine:
    """Runs a lineup of primary and tag detectors and returns a single primary per bar."""
    def __init__(
        self,
        cfg: Optional[RegimeConfig] = None,
        primary_order: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None
    ):
        self.cfg = cfg or RegimeConfig()
        # default ordering by priority
        if primary_order is None:
            self.primary_order = [k for k,_ in sorted(PRIMARY_REGISTRY.items(), key=lambda kv: -kv[1].priority)]
        else:
            self.primary_order = list(primary_order)
        self.tag_keys = list(tags) if tags is not None else list(TAG_REGISTRY.keys())

    def run(self, df: pd.DataFrame) -> RegimeResult:
        feats = compute_features(df, self.cfg)

        # 1) run all primaries
        prim_active: Dict[str, pd.Series] = {}
        prim_dir: Dict[str, pd.Series] = {}
        prim_conf: Dict[str, pd.Series] = {}

        for k in self.primary_order:
            prim = PRIMARY_REGISTRY[k]()
            act, d, conf = prim.detect(df, feats, self.cfg)
            prim_active[k] = act.astype(bool)
            prim_dir[k] = d.astype(float)
            prim_conf[k] = conf.astype(float)

        # 2) choose exactly one primary per bar (argmax by confidence; break ties by priority)
        idx = df.index
        primary_final = pd.Series([""]*len(idx), index=idx, name="primary")
        direction_final = pd.Series(np.zeros(len(idx)), index=idx, name="direction")
        confidence_final = pd.Series(np.zeros(len(idx)), index=idx, name="confidence")

        # Build confidence matrix (K x T)
        keys = self.primary_order
        conf_mat = np.vstack([prim_conf[k].values for k in keys])  # shape: K x T
        act_mat  = np.vstack([prim_active[k].values.astype(bool) for k in keys])
        dir_mat  = np.vstack([prim_dir[k].values for k in keys])

        # mask inactive
        conf_mat = np.where(act_mat, conf_mat, -1.0)  # inactive => -1 so never selected
        # argmax across primaries
        winner_idx = np.argmax(conf_mat, axis=0)  # index into keys
        # but if all -1 (no active), keep empty primary with 0 confidence
        no_active = np.max(conf_mat, axis=0) < 0

        for t in range(len(idx)):
            if no_active[t]:
                continue
            k = keys[winner_idx[t]]
            primary_final.iloc[t] = k
            direction_final.iloc[t] = dir_mat[winner_idx[t], t]
            confidence_final.iloc[t] = conf_mat[winner_idx[t], t]

        # 3) hysteresis: confirmation and exit
        primary_final = self._apply_hysteresis(primary_final, confidence_final)

        # 4) run tags (independent)
        tags_cols: Dict[str, pd.Series] = {}
        for k in self.tag_keys:
            tag = TAG_REGISTRY[k]()
            tags_cols[k] = tag.detect(df, feats, self.cfg).astype(bool)

        # pack tags into list per bar
        tag_list = []
        for t in range(len(idx)):
            present = [k for k in self.tag_keys if bool(tags_cols[k].iloc[t])]
            tag_list.append(present)

        regimes_df = pd.DataFrame({
            'primary': primary_final,
            'direction': direction_final,
            'confidence': confidence_final,
            'tags': tag_list
        }, index=idx)

        # include a few useful features directly in regimes_df for easy querying
        regimes_df['adx'] = feats['adx'].values
        regimes_df['atr_pct'] = feats['atr_pct'].values
        regimes_df['squeeze_on'] = feats['squeeze_on'].values.astype(bool)
        regimes_df['bb_width'] = feats['bb_width'].values

        return RegimeResult(regimes_df=regimes_df, features_df=feats)

    def _apply_hysteresis(self, primary: pd.Series, conf: pd.Series) -> pd.Series:
        """Require N confirm bars on entry, and N exit bars before clearing. Keeps direction/conf unchanged."""
        key_curr = ""
        confirm = 0
        exit_count = 0
        out = primary.copy()

        for i in range(len(primary)):
            k = primary.iloc[i]
            if k == "":
                # no active primary -> count exit
                exit_count += 1
                if exit_count >= self.cfg.exit_bars:
                    key_curr = ""
                out.iloc[i] = key_curr
                continue

            # new candidate
            if k != key_curr:
                confirm += 1
                if confirm >= self.cfg.confirm_bars:
                    key_curr = k
                    confirm = 0
                    exit_count = 0
                else:
                    # hold previous till confirmed
                    out.iloc[i] = key_curr
                    continue
            else:
                # staying in same regime
                confirm = 0
                exit_count = 0

            out.iloc[i] = key_curr

        return out

# ----------------------------
# Example: adding a new primary
# ----------------------------
# To add your own, subclass BasePrimaryDetector and decorate with @register_primary.
# Example skeleton:
#
# @register_primary
# class LiquidityCrisis(BasePrimaryDetector):
#     key = "liq_crisis"
#     priority = 90  # wins ties over others
#     def detect(self, df, feats, cfg):
#         # your logic -> active, direction, confidence
#         # e.g., extreme ATR%, huge gaps, etc.
#         active = (feats['atr_pct'].values >= 4.0)
#         direction = np.zeros_like(active, dtype=float)
#         conf = _normalize01(feats['atr_pct'].values - 4.0)
#         return _series_like(feats['close'], active.astype(bool), 'active'), \
#                _series_like(feats['close'], direction, 'dir'), \
#                _series_like(feats['close'], conf, 'conf')
#
# Same pattern for tags with @register_tag.

# ----------------------------
# Convenience runner
# ----------------------------
def annotate_regimes(df: pd.DataFrame, config: Optional[Mapping] = None) -> RegimeResult:
    """
    Quick one-liner to compute regimes with default detectors.
    Optional `config` can override RegimeConfig fields by dict.
    """
    cfg = RegimeConfig(**config) if config else RegimeConfig()
    engine = RegimeEngine(cfg=cfg)
    return engine.run(df)