// Pull the replay's inputs from a Home Assistant instance over its websocket API, on Node's built-in
// WebSocket: the production meter's hourly energy and the seven weather fields the integration
// archives. Writes data/production.json and data/weather.json next to this file.
//
//   HA_HOST=192.168.0.2:8123 HA_TOKEN=... PRODUCTION=sensor.your_meter_energy node tools/replay/fetch.mjs
//
// DAYS (default 420) sets the span. Long-term statistics are never purged, so the only real limit is
// how long the integration has been installed; a multi-year pull is what the long-memory variants in
// replay.py need. ENTRY_ID picks the config entry when the instance holds more than one.
import { haCall } from "./ha-ws.mjs";
import fs from "node:fs";
import path from "node:path";
const here = path.dirname(new URL(import.meta.url).pathname);
const production = process.env.PRODUCTION;
if (!process.env.HA_TOKEN || !production) { console.error("HA_TOKEN and PRODUCTION are required"); process.exit(1); }
const days = Number(process.env.DAYS ?? 420);
const end = new Date(); const start = new Date(end.getTime() - days * 86400e3);

// Half a year per request. One call for three years answers, but slowly and as a single enormous
// frame; chunking keeps every answer small and lets a partial pull still be diagnosed from the log.
const CHUNK_MS = 180 * 86400e3;
async function period(statistic_ids, types, units)
{
    const out = {};
    for (let t0 = start.getTime(); t0 < end.getTime(); t0 += CHUNK_MS)
    {
        const t1 = Math.min(end.getTime(), t0 + CHUNK_MS);
        const res = await haCall({
            type: "recorder/statistics_during_period", period: "hour", statistic_ids, types,
            start_time: new Date(t0).toISOString(), end_time: new Date(t1).toISOString(),
            ...(units ? { units } : {}),
        });
        for (const [id, rows] of Object.entries(res ?? {})) { (out[id] ??= []).push(...rows); }
    }
    return out;
}

// The seven archived weather fields, by the key the integration owns them under (statistics.py
// ARCHIVED_SERIES). Before 2026.9.5 the same series were written under entity ids, with three of
// them spelled differently; kept as a fallback so an old data pull still explains itself.
const FIELDS = ["cloud_cover", "ghi", "direct", "diffuse", "temperature", "wind_speed", "snow_depth"];
const LEGACY = {
    cloud_cover: "cloud_cover", ghi: "global_irradiance", direct: "direct_irradiance",
    diffuse: "diffuse_irradiance", temperature: "temperature", wind_speed: "wind_speed", snow_depth: "snow_depth",
};

// helios_forecast:<entry_id>_<key>. Ask the recorder what it holds rather than guessing the entry id.
const known = (await haCall({ type: "recorder/list_statistic_ids" }) ?? []).map((r) => r.statistic_id);
const entries = [...new Set(known.filter((id) => id.startsWith("helios_forecast:") && id.endsWith("_ghi"))
    .map((id) => id.slice("helios_forecast:".length, -"_ghi".length)))];
let ids, source;
if (entries.length === 0)
{
    source = "legacy entity ids (pre-2026.9.5 archive)";
    ids = Object.fromEntries(FIELDS.map((f) => [f, `sensor.helios_forecast_${LEGACY[f]}`]));
}
else
{
    const entry = process.env.ENTRY_ID ?? entries[0];
    if (!entries.includes(entry)) { console.error(`ENTRY_ID ${entry} not archived here. Known: ${entries.join(", ")}`); process.exit(1); }
    if (entries.length > 1 && !process.env.ENTRY_ID) { console.error(`note: ${entries.length} entries archived, using ${entry}. Set ENTRY_ID to pick another.`); }
    source = `helios_forecast:${entry}_*`;
    ids = Object.fromEntries(FIELDS.map((f) => [f, `helios_forecast:${entry}_${f}`]));
}

fs.mkdirSync(path.join(here, "data"), { recursive: true });

const prod = await period([production], ["change"], { energy: "kWh" });
const prows = (prod[production] ?? []).map((r) => [r.start, r.end, r.change]);
fs.writeFileSync(path.join(here, "data", "production.json"), JSON.stringify(prows));

const wx = await period(Object.values(ids), ["mean"]);
const byId = Object.fromEntries(Object.entries(ids).map(([field, id]) => [id, field]));
const out = {};
for (const [id, rows] of Object.entries(wx)) { out[byId[id] ?? id] = rows.map((r) => [r.start, r.mean]); }
fs.writeFileSync(path.join(here, "data", "weather.json"), JSON.stringify(out));

const span = (rows) => (rows.length ? `${new Date(rows[0][0]).toISOString().slice(0, 10)}..${new Date(rows.at(-1)[0]).toISOString().slice(0, 10)}` : "empty");
console.log(`weather from ${source}`);
console.log(`production: ${prows.length} hours ${span(prows)}`);
for (const f of FIELDS) { console.log(`  ${f.padEnd(12)} ${String(out[f]?.length ?? 0).padStart(6)} hours ${span(out[f] ?? [])}`); }
