# ============================================================
# CORE BOT LOGIC — FULL REWRITE
# Drop this section into app.py, replacing everything between
# "# ---------------- CORE BOT LOGIC ----------------" and
# "# ---------------- CHART DATA ----------------"
# No new dependencies required — uses only pandas (already imported).
# ============================================================


# ------------------------------------------------------------------ #
#  INDICATOR HELPERS                                                   #
# ------------------------------------------------------------------ #

def compute_ema(series, period):
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series, period=14):
    """RSI using Wilder's smoothing (EWM with com=period-1)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series, fast=12, slow=26, signal_period=9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df, period=14):
    """Average True Range — used for dynamic SL/TP and regime detection."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ------------------------------------------------------------------ #
#  SMART MONEY CONCEPTS (SMC) HELPERS                                  #
# ------------------------------------------------------------------ #

def compute_swing_points(df, lookback=5):
    """
    Returns (swing_highs, swing_lows) as lists of (index, price) tuples.
    A swing high is a candle whose 'high' is the highest in the window
    [i-lookback .. i+lookback]; same logic for swing lows.
    """
    highs = df["high"]
    lows  = df["low"]
    swing_highs, swing_lows = [], []

    for i in range(lookback, len(df) - lookback):
        window_h = highs.iloc[i - lookback : i + lookback + 1]
        window_l = lows.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window_h.max():
            swing_highs.append((i, float(highs.iloc[i])))
        if lows.iloc[i] == window_l.min():
            swing_lows.append((i, float(lows.iloc[i])))

    return swing_highs, swing_lows


def detect_bos(df, swing_highs, swing_lows):
    """
    Break of Structure:
      BOS_BULL — latest close breaks above the most recent swing high.
      BOS_BEAR — latest close breaks below the most recent swing low.
    Returns (bos_type | None, broken_level | None).
    """
    last_close = float(df.iloc[-1]["close"])

    if swing_highs and last_close > swing_highs[-1][1]:
        return "BOS_BULL", swing_highs[-1][1]

    if swing_lows and last_close < swing_lows[-1][1]:
        return "BOS_BEAR", swing_lows[-1][1]

    return None, None


def detect_order_blocks(df, lookback=20):
    """
    Bullish OB  — the last *bearish* candle in the lookback window
                  (price is likely to return here as support).
    Bearish OB  — the last *bullish* candle in the lookback window
                  (price is likely to return here as resistance).
    Returns dict with 'bullish' and 'bearish' keys (or None if not found).
    """
    obs = {"bullish": None, "bearish": None}
    if len(df) < 5:
        return obs

    start = max(0, len(df) - lookback)

    for i in range(len(df) - 2, start, -1):
        row = df.iloc[i]
        if obs["bullish"] is None and float(row["close"]) < float(row["open"]):
            obs["bullish"] = {
                "top":    float(row["open"]),
                "bottom": float(row["close"]),
                "index":  i,
                "time":   row["time"]
            }
        if obs["bearish"] is None and float(row["close"]) > float(row["open"]):
            obs["bearish"] = {
                "top":    float(row["close"]),
                "bottom": float(row["open"]),
                "index":  i,
                "time":   row["time"]
            }
        if obs["bullish"] and obs["bearish"]:
            break

    return obs


def detect_fvg(df, lookback=30):
    """
    Fair Value Gap (imbalance):
      Bullish FVG — candle[i-2].high < candle[i].low   (gap up, unmitigated demand)
      Bearish FVG — candle[i-2].low  > candle[i].high  (gap down, unmitigated supply)
    Returns the 5 most recent FVGs.
    """
    fvgs = []
    start = max(2, len(df) - lookback)

    for i in range(start, len(df)):
        c0_high = float(df.iloc[i - 2]["high"])
        c0_low  = float(df.iloc[i - 2]["low"])
        c2_high = float(df.iloc[i]["high"])
        c2_low  = float(df.iloc[i]["low"])

        if c2_low > c0_high:
            fvgs.append({"type": "bullish", "top": c2_low,  "bottom": c0_high, "index": i})
        elif c2_high < c0_low:
            fvgs.append({"type": "bearish", "top": c0_low,  "bottom": c2_high, "index": i})

    return fvgs[-5:] if fvgs else []


# ------------------------------------------------------------------ #
#  MARKET STRUCTURE & REGIME                                           #
# ------------------------------------------------------------------ #

def get_structure(df):
    """
    Uses EMA alignment (9 > 21 > 50) + sequential closes to classify
    the market as Bullish, Bearish, or Range.
    """
    if df is None or len(df) < 50:
        return "Range / Mixed"

    closes  = df["close"]
    ema9    = compute_ema(closes, 9).iloc[-1]
    ema21   = compute_ema(closes, 21).iloc[-1]
    ema50   = compute_ema(closes, 50).iloc[-1]
    c0, c1, c2 = closes.iloc[-1], closes.iloc[-3], closes.iloc[-6]

    if c0 > ema9 > ema21 > ema50 and c0 > c1 > c2:
        return "Bullish Structure"
    if c0 < ema9 < ema21 < ema50 and c0 < c1 < c2:
        return "Bearish Structure"
    return "Range / Mixed"


def get_market_regime(df):
    """
    Trending  — ATR% > 1.5 AND EMAs aligned.
    Active    — ATR% > 0.5.
    Range/Quiet — low volatility, no EMA alignment.
    """
    if df is None or len(df) < 20:
        return "Unknown"

    closes   = df["close"]
    atr_val  = float(compute_atr(df, 14).iloc[-1])
    price    = float(closes.iloc[-1]) or 1
    atr_pct  = (atr_val / price) * 100

    ema9  = float(compute_ema(closes, 9).iloc[-1])
    ema21 = float(compute_ema(closes, 21).iloc[-1])
    ema50 = float(compute_ema(closes, 50).iloc[-1]) if len(df) >= 50 else ema21

    trending = (ema9 > ema21 > ema50) or (ema9 < ema21 < ema50)

    if atr_pct > 1.5 and trending:
        return "Trending"
    if atr_pct > 0.5:
        return "Active"
    return "Range / Quiet"


# ------------------------------------------------------------------ #
#  SIGNAL GENERATION                                                   #
# ------------------------------------------------------------------ #

def generate_signal(df):
    """
    Core signal: EMA-9/21 crossover confirmed by MACD histogram direction
    and RSI not in an extreme zone.
    Returns 'BUY', 'SELL', or 'HOLD'.
    """
    if df is None or len(df) < 30:
        return "HOLD"

    closes = df["close"]
    ema9   = compute_ema(closes, 9)
    ema21  = compute_ema(closes, 21)
    _, _, histogram = compute_macd(closes)
    rsi    = compute_rsi(closes)

    ema9_now,  ema9_prev  = float(ema9.iloc[-1]),  float(ema9.iloc[-2])
    ema21_now, ema21_prev = float(ema21.iloc[-1]), float(ema21.iloc[-2])
    hist_now,  hist_prev  = float(histogram.iloc[-1]), float(histogram.iloc[-2])
    rsi_now               = float(rsi.iloc[-1])

    bull_cross = ema9_prev <= ema21_prev and ema9_now > ema21_now
    bear_cross = ema9_prev >= ema21_prev and ema9_now < ema21_now

    macd_bull = hist_now > 0 or (hist_prev < 0 and hist_now > hist_prev)
    macd_bear = hist_now < 0 or (hist_prev > 0 and hist_now < hist_prev)

    if bull_cross and macd_bull and 40 < rsi_now < 70:
        return "BUY"
    if bear_cross and macd_bear and 30 < rsi_now < 60:
        return "SELL"

    return "HOLD"


# ------------------------------------------------------------------ #
#  CONFIDENCE SCORING                                                  #
# ------------------------------------------------------------------ #

def estimate_confidence(df, signal):
    """
    Multi-factor scorer (max 95, min 35).

    Factor groups and max points:
      EMA alignment   → up to +20
      RSI position    → up to +8
      MACD momentum   → up to +12
      SMC: BOS        → +10
      SMC: FVG        → +7
      SMC: OB touch   → +8
      ATR activity    → +5
    """
    if df is None or len(df) < 30 or signal == "HOLD":
        return 50

    closes   = df["close"]
    price    = float(closes.iloc[-1]) or 1

    ema9     = compute_ema(closes, 9)
    ema21    = compute_ema(closes, 21)
    ema50    = compute_ema(closes, 50) if len(df) >= 50 else ema21
    _, _, histogram = compute_macd(closes)
    rsi      = compute_rsi(closes)
    atr      = compute_atr(df, 14)

    ema9_now   = float(ema9.iloc[-1])
    ema21_now  = float(ema21.iloc[-1])
    ema50_now  = float(ema50.iloc[-1])
    hist_now   = float(histogram.iloc[-1])
    hist_prev  = float(histogram.iloc[-2])
    rsi_now    = float(rsi.iloc[-1])
    atr_pct    = (float(atr.iloc[-1]) / price) * 100

    swing_highs, swing_lows = compute_swing_points(df)
    bos_type, _  = detect_bos(df, swing_highs, swing_lows)
    fvgs         = detect_fvg(df)
    obs          = detect_order_blocks(df)

    recent_cut   = len(df) - 10
    fvg_bull     = any(f["type"] == "bullish" and f["index"] > recent_cut for f in fvgs)
    fvg_bear     = any(f["type"] == "bearish" and f["index"] > recent_cut for f in fvgs)

    in_bull_ob = (
        obs["bullish"] is not None and
        obs["bullish"]["bottom"] <= price <= obs["bullish"]["top"]
    )
    in_bear_ob = (
        obs["bearish"] is not None and
        obs["bearish"]["bottom"] <= price <= obs["bearish"]["top"]
    )

    score = 50

    if signal == "BUY":
        # EMA alignment
        if ema9_now > ema21_now:  score += 8
        if ema21_now > ema50_now: score += 7
        if price > ema50_now:     score += 5
        # RSI
        if 45 <= rsi_now <= 65:   score += 8
        elif 35 <= rsi_now < 45:  score += 4
        # MACD
        if hist_now > 0:          score += 7
        if hist_now > hist_prev:  score += 5
        # SMC confluence
        if bos_type == "BOS_BULL": score += 10
        if fvg_bull:               score += 7
        if in_bull_ob:             score += 8
        # Volatility bonus
        if atr_pct > 0.5:         score += 5

    elif signal == "SELL":
        if ema9_now < ema21_now:  score += 8
        if ema21_now < ema50_now: score += 7
        if price < ema50_now:     score += 5
        if 35 <= rsi_now <= 55:   score += 8
        elif 55 < rsi_now <= 65:  score += 4
        if hist_now < 0:          score += 7
        if hist_now < hist_prev:  score += 5
        if bos_type == "BOS_BEAR": score += 10
        if fvg_bear:               score += 7
        if in_bear_ob:             score += 8
        if atr_pct > 0.5:         score += 5

    return max(35, min(95, score))


# ------------------------------------------------------------------ #
#  BIAS / TRADE IDEA HELPERS  (unchanged interface)                    #
# ------------------------------------------------------------------ #

def get_bias_from_signal(signal):
    if signal == "BUY":   return "Bullish"
    if signal == "SELL":  return "Bearish"
    return "Neutral"


def get_trade_idea(signal):
    if signal == "BUY":  return "Pullback long / continuation"
    if signal == "SELL": return "Reject highs / continuation short"
    return "Wait for clearer confirmation"


# ------------------------------------------------------------------ #
#  STRATEGY DISPATCHER  (evaluate_bot_window)                          #
# ------------------------------------------------------------------ #

def evaluate_bot_window(df, strategy="bot"):
    """
    Unified entry point for all strategies.

    Strategies:
      'basic'       — simple close > prev close direction
      'ema_rsi'     — EMA-9/21 cross + RSI gate (original, now improved)
      'smart_money' — SMC-first: BOS + FVG + OB confluence (≥3 of 5 factors)
      'bot'         — full confluence: generate_signal() + structure filter
                      + confidence gate (default)
    """
    if df is None or len(df) < 30:
        return {
            "signal":     "HOLD",
            "bias":       "Neutral",
            "structure":  "Range / Mixed",
            "regime":     "Unknown",
            "confidence": 50,
            "trade_idea": "Not enough data"
        }

    closes      = df["close"]
    latest      = float(closes.iloc[-1])
    prev        = float(closes.iloc[-2])
    latest_open = float(df.iloc[-1]["open"])
    latest_high = float(df.iloc[-1]["high"])

    structure = get_structure(df)
    regime    = get_market_regime(df)

    # Pre-compute shared indicators once
    ema9       = compute_ema(closes, 9)
    ema21      = compute_ema(closes, 21)
    _, _, hist = compute_macd(closes)
    rsi        = compute_rsi(closes)

    ema9_now,  ema9_prev  = float(ema9.iloc[-1]),  float(ema9.iloc[-2])
    ema21_now, ema21_prev = float(ema21.iloc[-1]), float(ema21.iloc[-2])
    hist_now              = float(hist.iloc[-1])
    rsi_now               = float(rsi.iloc[-1])

    final_signal = "HOLD"

    # ── basic ────────────────────────────────────────────────────────
    if strategy == "basic":
        if latest > prev:   final_signal = "BUY"
        elif latest < prev: final_signal = "SELL"

    # ── smart_money ──────────────────────────────────────────────────
    elif strategy == "smart_money":
        swing_highs, swing_lows = compute_swing_points(df)
        bos_type, _  = detect_bos(df, swing_highs, swing_lows)
        fvgs         = detect_fvg(df)
        obs          = detect_order_blocks(df)

        rc = len(df) - 10
        fvg_bull = any(f["type"] == "bullish" and f["index"] > rc for f in fvgs)
        fvg_bear = any(f["type"] == "bearish" and f["index"] > rc for f in fvgs)

        in_bull_ob = obs["bullish"] and obs["bullish"]["bottom"] <= latest <= obs["bullish"]["top"]
        in_bear_ob = obs["bearish"] and obs["bearish"]["bottom"] <= latest <= obs["bearish"]["top"]

        bull_score = sum([
            bos_type == "BOS_BULL",
            bool(fvg_bull),
            bool(in_bull_ob),
            ema9_now > ema21_now,
            rsi_now > 50
        ])
        bear_score = sum([
            bos_type == "BOS_BEAR",
            bool(fvg_bear),
            bool(in_bear_ob),
            ema9_now < ema21_now,
            rsi_now < 50
        ])

        if bull_score >= 3:   final_signal = "BUY"
        elif bear_score >= 3: final_signal = "SELL"

    # ── ema_rsi ───────────────────────────────────────────────────────
    elif strategy == "ema_rsi":
        bull_cross = ema9_prev <= ema21_prev and ema9_now > ema21_now
        bear_cross = ema9_prev >= ema21_prev and ema9_now < ema21_now

        if bull_cross and 40 < rsi_now < 70:  final_signal = "BUY"
        elif bear_cross and 30 < rsi_now < 60: final_signal = "SELL"

    # ── bot (default full-confluence) ────────────────────────────────
    else:
        raw_signal = generate_signal(df)
        if raw_signal == "BUY"  and structure != "Bearish Structure": final_signal = "BUY"
        elif raw_signal == "SELL" and structure != "Bullish Structure": final_signal = "SELL"

    # ── Confidence gate ───────────────────────────────────────────────
    confidence = estimate_confidence(df, final_signal)
    if final_signal != "HOLD" and confidence < bot_config["min_confidence"]:
        final_signal = "HOLD"

    # Fallback bias uses raw directional signal (even when gated to HOLD)
    raw_bias_signal = generate_signal(df)

    return {
        "signal":     final_signal,
        "bias":       get_bias_from_signal(final_signal) if final_signal != "HOLD"
                      else get_bias_from_signal(raw_bias_signal),
        "structure":  structure,
        "regime":     regime,
        "confidence": confidence,
        "trade_idea": get_trade_idea(final_signal)
    }


# ------------------------------------------------------------------ #
#  TRADE LEVELS  — ATR-based (replaces fixed % SL/TP)                 #
# ------------------------------------------------------------------ #

def calculate_trade_levels(df, signal):
    """
    SL placed at 1.5× ATR from entry.
    TP placed at SL-distance × risk_reward ratio.
    Much more adaptive than the previous fixed-percentage approach.
    """
    latest_close = float(df.iloc[-1]["close"])
    atr_val      = float(compute_atr(df, 14).iloc[-1])
    sl_distance  = atr_val * 1.5

    if signal == "BUY":
        sl = latest_close - sl_distance
        tp = latest_close + sl_distance * bot_config["risk_reward"]
        return {"entry": round(latest_close, 4),
                "sl":    round(sl, 4),
                "tp":    round(tp, 4)}

    if signal == "SELL":
        sl = latest_close + sl_distance
        tp = latest_close - sl_distance * bot_config["risk_reward"]
        return {"entry": round(latest_close, 4),
                "sl":    round(sl, 4),
                "tp":    round(tp, 4)}

    return {"entry": round(latest_close, 4),
            "sl":    round(latest_close, 4),
            "tp":    round(latest_close, 4)}
