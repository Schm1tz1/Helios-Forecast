# Replay

Re-forecast past days offline, as the integration would have forecast them that morning, and
score them against the meter. The residual map and the analog library are rebuilt from the
history available at 06:00 local; the archived weather stands in for the forecast weather, so the
weather error is out of the picture on purpose. What this measures is the learning.

    HA_HOST=192.168.0.2:8123 HA_TOKEN=... PRODUCTION=sensor.your_meter_energy node tools/replay/fetch.mjs
    .venv/bin/python tools/replay/replay.py --days 30

`replay.py` carries the installation's layout as constants; edit them for another roof. `data/`
stays out of git. Variants of the analog step live in `replay.py` so a change to the learning is
measured on sixty days before it is written into the integration.

`pipeline actuel` is the integration's own path (ratio analogs since 2026.9.3); `pipeline 2026.9.2 (watts)` keeps the previous watts library for comparison.

## Without a Home Assistant instance

    python3 tools/replay/synthetic.py && python3 tools/replay/replay.py --days 45

`synthetic.py` writes the same two files `fetch.mjs` does, for a roof that sits 10 % below the
generic model everywhere and loses all but 30 % of that inside one patch of sky (az 100-125,
alt 8-22) — a tree, in other words. The truth is known, so a variant claiming to learn near-field
shading can be checked: it has to move the bands the patch falls in and leave the others alone.
Delete `data/` before pulling real history again.

## What `fetch.mjs` reads

The production meter's hourly `change`, and the seven weather fields from the integration's own
archived statistics (`helios_forecast:<entry_id>_*`). Before 2026.9.5 those were written under
entity ids, three of them spelled differently; both layouts are handled, and `ENTRY_ID` picks the
entry when an instance holds more than one.

`DAYS` (default 420) sets the span, fetched in half-year chunks. Long-term statistics are never
purged, so the only real limit is how long the integration has been installed — which is what makes
the long-memory variants below measurable at all.

## Long memory (`memoire longue seule`, `memoire longue + pipeline`)

The integration rebuilds the residual map from a 60-day window every refresh, because that is all
Open-Meteo serves. Near-field shading is a property of the sun's angles, which recur annually, so a
60-day aperture relearns a fixed obstacle from scratch every year and arrives weeks late: at lat 50
the equinox horizon needs sun cells the learn window covers only 69 % of, with 12 of 39 confident.

`SkyAccumulator` in `replay.py` is the alternative, and it is written as the thing that would ship
rather than as a wider window:

- per sky cell, the sufficient statistics a `Store` would hold — a weight and a weighted sum of the
  ratio — decayed with a year's half-life instead of a month's;
- the stored ratio is **normalised by the site's global ratio at the time it was observed**, so the
  cell holds the shape of the sky and not the level. A panel wash moves the level, which stays on
  the integration's own 30-day recency, and leaves a year-old shadow where it was;
- each cell's weight saturates at `--cap`, which is what decides how many contradicting hours it
  takes to unlearn a tree that was cut down. Uncapped, a felled tree stays in the forecast for ever;
- folded forward once per bucket, never re-read, so the replay costs what a coordinator would cost.

`--half-life` (days, default 365) and `--cap` (default 40) are the two knobs; the run prints how many
sky cells each map holds, averaged per day, since that is the quantity the whole idea turns on.

On `synthetic.py`'s planted shadow, 45 days: the shaded morning bands go from 25 W/kWp / +93 % bias
to 19 / +68 %, and the band above from 45 / +30 % to 32 / +21 %, while the afternoon does not move.
173 sky cells confident enough to speak for themselves, against 46. Dropping `--half-life` to the
integration's own 30 days collapses both the accuracy back onto `residuel seul` and the cell count to
60, which is the check that the gain is the memory and not some other difference in the path.

The residual bias that survives is the map's own resolution: bilinear
sampling spreads one cell across its 2x2 neighbourhood, so the sharpest notch it can carve is about
20 degrees of azimuth wide.
