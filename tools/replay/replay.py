"""Replay the learning offline, day after day, against what the roof really produced.

Everything the forecast is made of is a pure function, so a day can be re-forecast as the
integration would have forecast it that morning: the residual map and the analog library are
rebuilt from the history available at 06:00 local, then the day is predicted and scored against the
meter. The archived weather stands in for the forecast weather, which removes the weather error
from the picture on purpose: what is measured here is the learning, not Open-Meteo.

Variants of the pipeline are scored side by side, by sun altitude, so a change to the learning is
measured on sixty days before it is written into the integration.

Usage: .venv/bin/python tools/replay/replay.py [--days 30]  (data from tools/replay/fetch.mjs)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from custom_components.helios_forecast import analog  # noqa: E402
from custom_components.helios_forecast.analog import AnalogSample, build_library, enrich_points  # noqa: E402
from custom_components.helios_forecast.forecast import ForecastPoint, build_forecast_series  # noqa: E402
from custom_components.helios_forecast.openmeteo import WeatherSeries  # noqa: E402
from custom_components.helios_forecast.solar.geometry import sun_position  # noqa: E402
from custom_components.helios_forecast.solar.irradiance import snow_cover_factor  # noqa: E402
from custom_components.helios_forecast.solar.power import (  # noqa: E402
    PanelOrientation, PvLayout, WeatherSample, compute_pv_power_per_array,
)
from custom_components.helios_forecast.solar import residual as RS  # noqa: E402
from custom_components.helios_forecast.solar.residual import (  # noqa: E402
    LEARN_DAYS, ProductionBucket, SkyResidualInput, SkyResidualMap, build_sky_residual_map,
)

HERE = os.path.dirname(__file__)
TZ = ZoneInfo("Europe/Paris")

# Jerome's installation, as declared in the integration (core.config_entries, 2026-09-06).
LAT, LON = 44.10974, 1.40441
LAYOUT = PvLayout(orientations=[PanelOrientation(tilt_deg=13.0, azimuth_deg=188.0, tracker=None)],
                  shares=[1.0], coords=[None], total_kwp=3.0, caps=[])
CAP_W = 3200.0


def load_weather() -> WeatherSeries:
    raw = json.load(open(os.path.join(HERE, "data", "weather.json")))
    by_t: Dict[int, Dict[str, float]] = {}
    for field, rows in raw.items():
        for start_ms, mean in rows:
            if mean is not None:
                by_t.setdefault(int(start_ms), {})[field] = float(mean)
    times = sorted(by_t)

    def pick(t: int, field: str, legacy: str, default: float) -> float:
        """The archive's own key since 2026.9.5, falling back to the entity-id spelling three of the
        fields used before the migration, so a data/ pulled by an older fetch.mjs still loads."""
        row = by_t[t]
        value = row.get(field, row.get(legacy))
        return float(value) if value is not None else default

    return WeatherSeries(
        times=[datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in times],
        cloud=[pick(t, "cloud_cover", "cloud_cover", 0.0) for t in times],
        shortwave=[pick(t, "ghi", "global_irradiance", 0.0) for t in times],
        direct=[pick(t, "direct", "direct_irradiance", 0.0) for t in times],
        diffuse=[pick(t, "diffuse", "diffuse_irradiance", 0.0) for t in times],
        temp=[pick(t, "temperature", "temperature", 20.0) for t in times],
        wind=[pick(t, "wind_speed", "wind_speed", 0.0) for t in times],
        snow=[pick(t, "snow_depth", "snow_depth", 0.0) for t in times],
    )


def load_production() -> List[ProductionBucket]:
    rows = json.load(open(os.path.join(HERE, "data", "production.json")))
    out = []
    for start_ms, end_ms, change in rows:
        if change is None or not math.isfinite(change) or change < 0:
            continue  # a meter reset reads as a huge negative hour; the integration skips it too
        out.append(ProductionBucket(start_ms=float(start_ms), end_ms=float(end_ms), kwh=float(change)))
    out.sort(key=lambda b: b.start_ms)  # the accumulator below folds in time order and never looks back
    return out


def measured_map(buckets: List[ProductionBucket]) -> Dict[int, float]:
    """kWh per local hour, keyed on the epoch ms of the hour's start."""
    return {int(b.start_ms): b.kwh for b in buckets}


def physics_w(moment: datetime, weather: WeatherSeries, epochs: List[float]) -> float:
    """Bare physical watts at an instant, from the archived weather, the way the residual map sees it."""
    ms = moment.timestamp() * 1000
    i = max(0, min(len(epochs) - 1, _nearest(epochs, ms)))
    sample = WeatherSample(cloud=weather.cloud[i] or 0.0, ghi=weather.shortwave[i], direct=weather.direct[i],
                           diffuse=weather.diffuse[i], temp=weather.temp[i], wind=weather.wind[i])
    pcts = compute_pv_power_per_array(moment, LAT, LON, sample, LAYOUT)
    return max(0.0, min(CAP_W, pcts[0] * LAYOUT.total_kwp * 10.0))


def _nearest(epochs: List[float], ms: float) -> int:
    lo, hi = 0, len(epochs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if epochs[mid] <= ms:
            lo = mid
        else:
            hi = mid
    return lo if abs(epochs[lo] - ms) <= abs(epochs[hi] - ms) else hi


# ---- variants of the analog step ------------------------------------------------------------------

def predict_variant(library, alt, az, cloud, temp, *, narrow_alt=False, az_weight=None, az_norm=180.0):
    """analog.predict with switches: altitude distance relative to the altitude itself (a low sun only
    matches a low sun), and a heavier, tighter azimuth term so a September evening at 271 degrees is not
    told by July evenings at 290 that lay behind a different piece of horizon."""
    if not library or alt <= 0:
        return None
    w_az = analog._W_AZ if az_weight is None else az_weight
    scored = []
    for s in library:
        dalt = (s.alt - alt) / (max(alt, 10.0) if narrow_alt else 90.0)
        daz = analog._az_diff(s.az, az) / az_norm
        dcl = (s.cloud - cloud) / 100.0
        d2 = analog._W_ALT * dalt * dalt + w_az * daz * daz + analog._W_CLOUD * dcl * dcl
        if temp is not None and s.temp is not None:
            dt = (s.temp - temp) / analog._TEMP_SCALE
            d2 += analog._W_TEMP * dt * dt
        else:
            d2 += analog._TEMP_MISSING_PENALTY
        scored.append((d2, s.watt))
    scored.sort(key=lambda x: x[0])
    top = scored[: analog._K]
    weighted = [(v, math.exp(-d2 / (2.0 * analog._BANDWIDTH2))) for d2, v in top]
    p10, p50, p90 = analog._weighted_percentiles(weighted, (0.10, 0.50, 0.90))
    n_close = sum(1 for d2, _ in top if d2 <= analog._CLOSE_D2)
    confidence = min(1.0, n_close / analog._CONFIDENCE_FULL)
    ceiling = p90 * analog._CEILING_MARGIN if n_close >= analog._CEILING_MIN_ANALOGS else None
    return analog.AnalogBand(p10=p10, p50=p50, p90=p90, confidence=confidence, ceiling=ceiling)


def enrich_variant(points, library, weather, w_epochs, now, *, ratio_lib=False, narrow_alt=False, max_conf=1.0,
                   az_weight=None, az_norm=180.0):
    out = []
    for p in points:
        if p.t < now:
            out.append(p)
            continue
        sun = sun_position(p.t, LAT, LON)
        if sun.altitude <= 0:
            out.append(replace(p, pv_p10=0.0, pv_p90=0.0))
            continue
        ms = p.t.timestamp() * 1000.0
        cloud = analog._sample_series(weather.times, weather.cloud, ms, w_epochs)
        temp = analog._sample_series(weather.times, weather.temp, ms, w_epochs)
        band = predict_variant(library, sun.altitude, sun.azimuth, cloud if cloud is not None else 50.0, temp,
                               narrow_alt=narrow_alt, az_weight=az_weight, az_norm=az_norm)
        if band is None:
            out.append(p)
            continue
        c = min(max_conf, band.confidence)
        if ratio_lib:
            # The library holds actual/physics ratios: the analog's word is a ratio applied to today's physics.
            p50, p10, p90 = band.p50 * p.pv_raw_w, band.p10 * p.pv_raw_w, band.p90 * p.pv_raw_w
            ceiling = band.ceiling * p.pv_raw_w if band.ceiling is not None else None
        else:
            p50, p10, p90, ceiling = band.p50, band.p10, band.p90, band.ceiling
        blended = c * p50 + (1.0 - c) * p.pv_w
        if ceiling is not None:
            blended = min(blended, ceiling)
        out.append(replace(p, pv_w=blended, pv_p10=p10, pv_p90=p90) if band.confidence >= analog.BAND_MIN_CONFIDENCE
                   else replace(p, pv_w=blended))
    return out


def ratio_library(buckets, weather, epochs) -> List[AnalogSample]:
    """The analog library with each sample's watts replaced by its ratio to the physics of that hour."""
    lib = build_library(buckets, weather, LAT, LON)
    out = []
    it = iter(lib)
    for b in buckets:
        mid = datetime.fromtimestamp((b.start_ms + b.end_ms) / 2000.0, tz=timezone.utc)
        if sun_position(mid, LAT, LON).altitude <= 0 or not math.isfinite(b.kwh):
            continue
        s = next(it, None)
        if s is None:
            break
        model = physics_w(mid, weather, epochs)
        if model < 30.0:
            continue  # a ratio against nothing says nothing
        out.append(AnalogSample(alt=s.alt, az=s.az, cloud=s.cloud, watt=s.watt / model, temp=s.temp))
    return out


# ---- a variant of the residual map's smoothing ---------------------------------------------------

def vertical_map(inp: SkyResidualInput) -> Optional[SkyResidualMap]:
    """The residual map with its thin cells filled from the sky ABOVE and BELOW them at the same azimuth
    first, and from the side cells only when that column is empty.

    Near-field shading is a property of azimuth: an obstacle at 288 degrees darkens every cell under its
    own height at 288 and nothing at 271. The integration's smoothing pulls a thin cell toward all eight
    neighbours alike, so a cell the sun only reaches this month borrows the shading of the azimuths it
    reached last month. The column above it is a better witness."""
    keep = RS.SMOOTH_NEIGHBOR_W
    # Not zero: the integration's own smoothing divides by the pull, so an all-but-absent pull leaves the
    # raw cells untouched without tripping it.
    RS.SMOOTH_NEIGHBOR_W = 1e-9
    try:
        raw = build_sky_residual_map(inp)
    finally:
        RS.SMOOTH_NEIGHBOR_W = keep
    if raw is None:
        return None
    m, conf, n_az, n_alt = list(raw.m), list(raw.conf), raw.n_az, raw.n_alt
    m_s, c_s = list(m), list(conf)
    for aj in range(n_alt):
        for ai in range(n_az):
            idx = aj * n_az + ai
            if conf[idx] >= RS.SMOOTH_CONF_KEEP:
                continue
            num = den = best = 0.0
            for d_alt in (-1, 1, -2, 2):
                nj = aj + d_alt
                if 0 <= nj < n_alt:
                    c = conf[nj * n_az + ai] / abs(d_alt)
                    num += c * m[nj * n_az + ai]
                    den += c
                    best = max(best, conf[nj * n_az + ai])
            if den <= 0:
                for d_az in (-1, 1):
                    ni = (ai + d_az) % n_az
                    c = conf[aj * n_az + ni]
                    num += c * m[aj * n_az + ni]
                    den += c
                    best = max(best, c)
            if den <= 0 or best <= 0:
                continue
            pull = keep * best
            m_s[idx] = (conf[idx] * m[idx] + pull * (num / den)) / (conf[idx] + pull)
            c_s[idx] = min(1.0, conf[idx] + pull)
    return replace(raw, m=m_s, conf=c_s)


# ---- a variant of where the map's memory lives ---------------------------------------------------
#
# The integration rebuilds the residual map from a 60-day window on every refresh, because that is
# all Open-Meteo serves. But near-field shading is a property of the sun's angles, which recur every
# year, and the archived weather is not capped at 60 days: the integration writes it to its own
# long-term statistics precisely so the history grows without bound. Nothing reads it back yet.
#
# What is measured here is whether a map that REMEMBERS pays: per sky cell, the sufficient statistics
# a Store would hold (a weight and a weighted sum), decayed with a half-life of a year instead of a
# month, folded forward once per bucket and never re-read. Three properties make it more than a
# wider window:
#
#   - the stored ratio is NORMALISED by the site's global ratio at the time it was observed, so it
#     carries the SHAPE of the sky and not the level. A panel wash, soiling or a thermal-model
#     correction moves the level, which stays on the integration's own 30-day recency, and leaves a
#     year-old shadow where it was.
#   - each cell's weight saturates at CELL_W_MAX, which is what decides how many contradicting hours
#     it takes to unlearn a tree that was cut down. Uncapped, the memory is unbounded and a felled
#     tree stays in the forecast for ever.
#   - it is folded incrementally, in time order, exactly as a coordinator would fold one refresh:
#     O(1) work per new hour, no re-reading of history. The replay does the same so what it measures
#     is what would run.

LONG_HALF_LIFE_DAYS = 365.0
CELL_W_MAX = 40.0


def model_kwh_of(b: ProductionBucket, weather: WeatherSeries, epochs: List[float]) -> Optional[float]:
    """The physical model's kWh over one bucket, mirroring build_sky_residual_map: the nearest weather
    hour, four sub-samples, per-array caps then the entry cap, averaged over the WHOLE bucket."""
    mid = (b.start_ms + b.end_ms) / 2.0
    i = _nearest(epochs, mid)
    cloud = weather.cloud[i]
    cloud = max(0.0, min(100.0, cloud)) if (cloud is not None and math.isfinite(cloud)) else 0.0
    sample = WeatherSample(cloud=cloud, ghi=weather.shortwave[i], direct=weather.direct[i],
                           diffuse=weather.diffuse[i], temp=weather.temp[i], wind=weather.wind[i])
    snow = snow_cover_factor(weather.snow[i], sample.temp)
    k = LAYOUT.total_kwp * 10.0
    total = 0.0
    seen = 0
    for j in range(RS.LEARN_SUBSAMPLES):
        sub = b.start_ms + (j + 0.5) * (b.end_ms - b.start_ms) / RS.LEARN_SUBSAMPLES
        moment = datetime.fromtimestamp(sub / 1000.0, tz=timezone.utc)
        if sun_position(moment, LAT, LON).altitude <= 0:
            continue
        pcts = compute_pv_power_per_array(moment, LAT, LON, sample, LAYOUT)
        total += min(CAP_W, RS.capped_model_kwh(pcts, LAYOUT, k, snow) * 1000.0) / 1000.0
        seen += 1
    return total / RS.LEARN_SUBSAMPLES if seen else None


class SkyAccumulator:
    """Per sky cell: the weight and the weighted sum of the normalised ratio, plus the instant each
    cell was last folded (the decay is applied lazily, per cell, on the next touch)."""

    def __init__(self, half_life_days: float = LONG_HALF_LIFE_DAYS, w_max: float = CELL_W_MAX) -> None:
        n = RS.N_AZ * RS.N_ALT
        self.w = [0.0] * n
        self.wr = [0.0] * n
        self.t = [0.0] * n
        self.hl_ms = half_life_days * 86400e3
        self.w_max = w_max
        # The running site level: the integration's own 30-day half-life, energy-weighted the way
        # build_sky_residual_map weights its global fallback. Causal by construction, since folding
        # only ever moves forwards.
        self.g_w = 0.0
        self.g_wr = 0.0
        self.g_t = 0.0

    @property
    def global_ratio(self) -> float:
        return self.g_wr / self.g_w if self.g_w > 0 else 1.0

    def fold(self, b: ProductionBucket, weather: WeatherSeries, epochs: List[float]) -> None:
        """Fold one measured hour in. Every guard is build_sky_residual_map's, in the same order."""
        kwh = b.kwh
        if not math.isfinite(kwh) or kwh < 0:
            return
        if RS.implausible_hour(kwh, b.start_ms, b.end_ms, LAYOUT.total_kwp):
            return
        mid = (b.start_ms + b.end_ms) / 2.0
        sun = sun_position(datetime.fromtimestamp(mid / 1000.0, tz=timezone.utc), LAT, LON)
        if sun.altitude <= 0:
            return
        model = model_kwh_of(b, weather, epochs)
        if model is None or model < RS.MODEL_KWH_FLOOR:
            return
        if b.curtailed and kwh < model:
            return  # right-censored: the inverter set this figure, not the sky
        ratio = kwh / model
        if not math.isfinite(ratio) or ratio < 0:
            return

        i = _nearest(epochs, mid)
        cloud = weather.cloud[i]
        cloud = max(0.0, min(100.0, cloud)) if (cloud is not None and math.isfinite(cloud)) else 0.0
        clear = max(0.1, 1 - cloud / 100)

        # Level first, so the shape stored below is measured against the site as it was that week.
        decay_g = math.pow(0.5, max(0.0, mid - self.g_t) / RS.RECENCY_HALF_LIFE_MS) if self.g_t else 1.0
        self.g_w = self.g_w * decay_g + clear * model
        self.g_wr = self.g_wr * decay_g + clear * model * ratio
        self.g_t = mid
        g = self.global_ratio
        if not (g > 0):
            return

        idx = (min(RS.N_ALT - 1, max(0, math.floor(sun.altitude / RS.ALT_STEP_DEG))) * RS.N_AZ
               + min(RS.N_AZ - 1, max(0, math.floor((sun.azimuth % 360) / RS.AZ_STEP_DEG))))
        decay = math.pow(0.5, max(0.0, mid - self.t[idx]) / self.hl_ms) if self.t[idx] else 1.0
        w = self.w[idx] * decay + clear
        wr = self.wr[idx] * decay + clear * (ratio / g)
        if w > self.w_max:
            # Both sides scaled together: the cap bounds the memory without disturbing the mean it holds.
            wr *= self.w_max / w
            w = self.w_max
        self.w[idx], self.wr[idx], self.t[idx] = w, wr, mid

    def visited(self) -> int:
        return sum(1 for w in self.w if w > 0)


def _smooth(m: List[float], conf: List[float]) -> tuple:
    """build_sky_residual_map's own thin-cell smoothing, lifted so the long map is judged on the same
    terms as the short one rather than on the absence of a step the integration performs."""
    m_s, c_s = list(m), list(conf)
    for aj in range(RS.N_ALT):
        for ai in range(RS.N_AZ):
            idx = aj * RS.N_AZ + ai
            self_conf = conf[idx]
            if self_conf >= RS.SMOOTH_CONF_KEEP:
                continue
            num = den = best = 0.0
            for d_alt in (-1, 0, 1):
                nj = aj + d_alt
                if nj < 0 or nj >= RS.N_ALT:
                    continue
                for d_az in (-1, 0, 1):
                    if d_alt == 0 and d_az == 0:
                        continue
                    n_idx = nj * RS.N_AZ + (ai + d_az + RS.N_AZ) % RS.N_AZ
                    c = conf[n_idx]
                    num += c * m[n_idx]
                    den += c
                    best = max(best, c)
            if den <= 0 or best <= 0:
                continue
            pull = RS.SMOOTH_NEIGHBOR_W * best
            m_s[idx] = (self_conf * m[idx] + pull * (num / den)) / (self_conf + pull)
            c_s[idx] = min(1.0, self_conf + pull)
    return m_s, c_s


def long_map(acc: SkyAccumulator, now_ms: float, short: Optional[SkyResidualMap]) -> Optional[SkyResidualMap]:
    """The accumulator read as a SkyResidualMap: the remembered SHAPE, scaled by the level the short
    60-day window reports today. wr/w is decay-invariant, so only the confidence needs bringing to now."""
    g = short.global_ratio if short is not None else acc.global_ratio
    if not (g > 0):
        return None
    n = RS.N_AZ * RS.N_ALT
    m = [g] * n
    conf = [0.0] * n
    visited = 0
    total_w = 0.0
    for i in range(n):
        if acc.w[i] <= 0:
            continue
        m[i] = max(RS.M_MIN, min(RS.M_MAX, acc.wr[i] / acc.w[i] * g))
        w_now = acc.w[i] * math.pow(0.5, max(0.0, now_ms - acc.t[i]) / acc.hl_ms)
        conf[i] = 1 - math.exp(-w_now / RS.CONF_W0)
        total_w += w_now
        visited += 1
    if total_w < RS.MIN_TOTAL_WEIGHT:
        return None
    m_s, c_s = _smooth(m, conf)
    return SkyResidualMap(n_az=RS.N_AZ, n_alt=RS.N_ALT, m=m_s, conf=c_s, global_ratio=g,
                          total_weight=total_w, visited_cells=visited)


# ---- the replay ------------------------------------------------------------------------------------

BANDS = ["AM 0-15", "AM 15-30", "AM 30+", "PM 30+", "PM 15-30", "PM 0-15"]


def band_of(sun) -> str:
    half = "AM" if sun.azimuth < 180 else "PM"
    alt = sun.altitude
    return "%s %s" % (half, "0-15" if alt < 15 else ("15-30" if alt < 30 else "30+"))


def replay(days: int, half_life: float = LONG_HALF_LIFE_DAYS, cap: float = CELL_W_MAX) -> None:
    weather = load_weather()
    epochs = [t.timestamp() * 1000.0 for t in weather.times]
    buckets = load_production()
    measured = measured_map(buckets)
    last_full = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    first = last_full - timedelta(days=days)

    VARIANTS = ["physique", "residuel seul", "pipeline actuel", "pipeline 2026.9.2 (watts)", "analogues azimut pese", "analogues sur ratios",
                "ratios + azimut pese", "lissage vertical seul", "vertical + pipeline", "vertical + ratios + azimut",
                "memoire longue seule", "memoire longue + pipeline"]
    acc_stats = {v: {b: {"n": 0, "abs": 0.0, "pred": 0.0, "truth": 0.0} for b in BANDS + ["total"]} for v in VARIANTS}
    daily = {v: [] for v in VARIANTS}
    band_cov = {v: [0, 0] for v in VARIANTS}
    w_epochs = analog.series_epochs(weather.times)

    # One accumulator for the whole run, folded forward as the days go by: each measured hour is
    # folded exactly once, the way a coordinator would fold it at the refresh that first saw it.
    # Everything before the first replayed day is the warm-up, and it is free: those hours would
    # have been folded then too.
    acc = SkyAccumulator(half_life_days=half_life, w_max=cap)
    folded = 0
    coverage = [0, 0]  # cells the long map holds vs cells the 60-day map holds, summed over the days
    n_days = 0

    day = first
    while day < last_full:
        now = day.replace(hour=6)
        now_ms = now.timestamp() * 1000.0
        history = [b for b in buckets if b.end_ms <= now_ms and b.start_ms >= (now - timedelta(days=LEARN_DAYS)).timestamp() * 1000]
        if len(history) < 24 * 10:
            day += timedelta(days=1)
            continue
        inp = SkyResidualInput(
            lat=LAT, lon=LON, layout=LAYOUT, production=history, inverter_max_w=CAP_W,
            cloud_times=epochs, cloud=weather.cloud,
            shortwave=weather.shortwave, direct=weather.direct, diffuse=weather.diffuse, temp=weather.temp,
            wind=weather.wind, snow=weather.snow, now_ms=now_ms)
        sky = build_sky_residual_map(inp)
        vsky = vertical_map(inp)
        while folded < len(buckets) and buckets[folded].end_ms <= now_ms:
            acc.fold(buckets[folded], weather, epochs)
            folded += 1
        lsky = long_map(acc, now_ms, sky)
        if sky is not None:
            # Cells confident enough to speak for themselves. A cell merely VISITED says little: below
            # SMOOTH_CONF_KEEP the smoothing overwrites it with its neighbours, which is exactly what
            # erases a shadow, so counting visits would report a memory the map does not actually have.
            firm = lambda mp: sum(1 for c in mp.conf if c >= RS.SMOOTH_CONF_KEEP)  # noqa: E731
            coverage[0] += firm(lsky) if lsky is not None else 0
            coverage[1] += firm(sky)
        start, end = day, day + timedelta(days=1)
        base = build_forecast_series(weather, LAYOUT, LAT, LON, inverter_max_w=CAP_W, start=start, end=end,
                                     step_minutes=15, residual_map=sky)
        vbase = build_forecast_series(weather, LAYOUT, LAT, LON, inverter_max_w=CAP_W, start=start, end=end,
                                      step_minutes=15, residual_map=vsky)
        lbase = build_forecast_series(weather, LAYOUT, LAT, LON, inverter_max_w=CAP_W, start=start, end=end,
                                      step_minutes=15, residual_map=lsky)
        library = build_library(history, weather, LAT, LON, LAYOUT, CAP_W)  # the integration's own call, ratios included
        watts_library = build_library(history, weather, LAT, LON)  # the pre-2026.9.3 library, watts only
        rlib = ratio_library(history, weather, epochs)
        AZ = dict(az_weight=0.7, az_norm=90.0)
        series = {
            "physique": [replace(p, pv_w=p.pv_raw_w) for p in base],
            "residuel seul": base,
            "pipeline actuel": enrich_points(base, library, weather, LAT, LON, now),
            "pipeline 2026.9.2 (watts)": enrich_points(base, watts_library, weather, LAT, LON, now),
            "analogues azimut pese": enrich_variant(base, watts_library, weather, w_epochs, now, **AZ),
            "analogues sur ratios": enrich_variant(base, rlib, weather, w_epochs, now, ratio_lib=True),
            "ratios + azimut pese": enrich_variant(base, rlib, weather, w_epochs, now, ratio_lib=True, **AZ),
            "lissage vertical seul": vbase,
            "vertical + pipeline": enrich_points(vbase, library, weather, LAT, LON, now),
            "vertical + ratios + azimut": enrich_variant(vbase, rlib, weather, w_epochs, now, ratio_lib=True, **AZ),
            "memoire longue seule": lbase,
            "memoire longue + pipeline": enrich_points(lbase, library, weather, LAT, LON, now),
        }
        for name, pts in series.items():
            by_hour: Dict[int, List[ForecastPoint]] = {}
            for p in pts:
                by_hour.setdefault(int(p.t.replace(minute=0, second=0, microsecond=0).timestamp() * 1000), []).append(p)
            day_pred = day_truth = 0.0
            for hour_ms, group in by_hour.items():
                truth = measured.get(hour_ms)
                mid = datetime.fromtimestamp(hour_ms / 1000 + 1800, tz=timezone.utc)
                sun = sun_position(mid, LAT, LON)
                if truth is None or truth <= 0 or sun.altitude <= 0 or mid < now:
                    continue
                pred_w = sum(p.pv_w for p in group) / len(group)
                truth_w = truth * 1000.0
                for key in (band_of(sun), "total"):
                    a = acc_stats[name][key]
                    a["n"] += 1; a["abs"] += abs(pred_w - truth_w); a["pred"] += pred_w; a["truth"] += truth_w
                day_pred += pred_w / 1000; day_truth += truth
                lo = [p.pv_p10 for p in group if p.pv_p10 is not None]
                hi = [p.pv_p90 for p in group if p.pv_p90 is not None]
                if lo and hi:
                    band_cov[name][1] += 1
                    band_cov[name][0] += (min(lo) <= truth_w <= max(hi))
            if day_truth > 0.3:
                daily[name].append((day_pred - day_truth) / day_truth * 100)
        n_days += 1
        day += timedelta(days=1)

    n_days = max(1, n_days)
    print("Replay on %d days, hourly, sun up, meter above zero. Per cell: nMAE (W/kWc) and bias; last column: mean absolute error on the day's energy." % days)
    print("%-27s" % "" + "".join("%11s" % b for b in BANDS) + "%11s %8s" % ("TOTAL", "jour"))
    for v in VARIANTS:
        cells = []
        for b in BANDS + ["total"]:
            a = acc_stats[v][b]
            cells.append("%3.0f %+4.0f%%" % (a["abs"] / a["n"] / LAYOUT.total_kwp, 100 * (a["pred"] - a["truth"]) / a["truth"]) if a["n"] else "    -    ")
        d = daily[v]
        mape = sum(abs(x) for x in d) / len(d) if d else float("nan")
        print("%-27s" % v + "".join("%11s" % c for c in cells) + "%8s" % ("%.1f%%" % mape))
    print("  n par cellule : " + " ".join("%s=%d" % (b, acc_stats["physique"][b]["n"]) for b in BANDS))
    if coverage[1]:
        print("\nCellules du ciel sures d'elles (conf >= %.1f), moyenne par jour : memoire longue %.0f, "
              "fenetre 60 jours %.0f (demi-vie %.0f j, plafond %.0f)."
              % (RS.SMOOTH_CONF_KEEP, coverage[0] / n_days, coverage[1] / n_days, half_life, cap))
    print("\nCouverture de la fourchette P10-P90 (objectif 80 %) :")
    for v in VARIANTS:
        hit, n = band_cov[v]
        if n:
            print("  %-27s %5.1f%%  (n=%d)" % (v, 100 * hit / n, n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--half-life", type=float, default=LONG_HALF_LIFE_DAYS,
                    help="the long map's per-cell half-life in days (the integration's own is 30)")
    ap.add_argument("--cap", type=float, default=CELL_W_MAX,
                    help="per-cell weight ceiling: how fast a felled tree can be unlearned")
    args = ap.parse_args()
    replay(args.days, args.half_life, args.cap)
