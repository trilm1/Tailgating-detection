# Dataset

The initial group contains 5 clips recorded at the same door, at 854×480 resolution and
approximately 24.9 FPS. An IMOU camera is mounted above the door frame and angled downward.
The left side of the image shows the outside courtyard, and the right side shows the room.
People standing near the threshold often have their feet cut off at the bottom of the image.

## File Format

Each clip in `clips/` includes:

- `<clip>.mp4`
- `<clip>_grants.csv`: `grant_id,t_unlock_s,t_relock_s`
- `<clip>_gt.csv`: `person_id,t_cross_s,direction,grant_id`. Each row represents one threshold crossing, either `in` or `out`. Leave `grant_id` empty when there is no grant.
- `<clip>_scenarios.json`: grant ID → scenario number (1–10), used for the confusion table.

Runtime configurations are stored separately in `configs/<clip>.yaml`, with one file for
each of the nine clips. The filename stem matches the video and its labels. Any
`<clip>_annotations.json` files in `clips/` are annotation-tool project files, not runtime configs.

**Grants are simulated.** The first person touches a phone to the door frame to simulate a
credential tap. Unlock/relock timestamps are annotated after recording; relock is placed
immediately after the last person in that event.

## Clip Inventory

| Clip | Duration | Grants | In / Out | Scenarios (grant counts) | Lighting and Conditions |
|---|---:|---:|---:|---|---|
| 1791534560789… | 143 s | 4 | 5 / 3 | 1 (2), 5 (2) | Daylight; courtyard brighter than the room |
| 1791534560791… | 67 s | 6 | 7 / 3 | 1 (6) | Same as above |
| 1791534560797… | 128 s | 6 | 9 / 7 | 1 (3), 2 (3) | Same as above |
| 1791539926064… | 363 s | 26 | 35 / 35 | 1 (18), 2 (7), 4 (1) | Camera changes color mode at 20 s; room overexposed around 30–45 s; dogs moving nearby |
| 1791539947261… | 323 s | 19 | 21 / 19 | 1 (11), 4 (1), 5 (1), 7 (4), 8 (2) | Green color cast; color mode changes at 275 s; dogs present; a person carrying a pot |

Camera height: <to be filled in>.

### Additions on 11 October 2026

The 4 new clips use their individual `configs/<clip>.yaml` files with `footprint` mode.
Their metadata reports approximately 14.27 FPS.
Camera height and detailed recording conditions: not yet filled in.

| Clip | Resolution | Grants | Scenarios (grant counts) |
|---|---|---:|---|
| 1791693967463… | 864×480 | 4 | 2 (3), 1 (1) |
| 1791694036909… | 868×480 | 4 | 3 (4); person IDs for g01 need confirmation |
| 1791694085906… | 856×480 | 4 | 8 (3), 1 (1) |
| 1791694149613… | 862×480 | 0 | Scenario 6: walking past without crossing the threshold |

For …6909/g01, two `in` rows share `p01`, so GT counts one person. If these are two different
people, they need separate IDs before the metrics are finalized.

Clip …9613 tests people walking past the door without entering. There are no grants or
crossings, so both CSV files contain only their headers. The system produces
`events.json = []` with no alerts, as expected. The number of walk-past events and the time
interval of each event have not yet been documented. These events are counted separately
for scenario 6 and excluded from the count-accuracy denominator and grant-level confusion statistics.

## Scenario Coverage

| # | Scenario | Recorded Events / Minimum |
|---|---|---|
| 1 | One person enters | 42 / 10 |
| 2 | A second person follows closely (< 1.5 s) | 13 / 6 |
| 3 | Two people enter side by side or overlapping | 4 / 4 |
| 4 | Three or more people enter | 2 / 2 |
| 5 | One person exits while another enters | 3 / 3 |
| 6 | A person walks past without entering | 4 / 4 |
| 7 | A person hesitates or steps back, then enters | 4 / 3 |
| 8 | A person carries a large object | 5 / 3 |
| 9 | Entry without a grant | 2 / 2 |
| 10 | Poor conditions | 3 / 3 |
