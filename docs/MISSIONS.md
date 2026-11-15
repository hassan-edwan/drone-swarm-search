# Missions

Every mission is a field scenario drawn from its family with all effects
randomised (`sample_scenario`), then one setting fixed. That way, a change in
results can be traced to one cause. Seeds are shared across all missions, so
mission *k* of two conditions starts from the same draw.
Register IDs refer to [ASSUMPTIONS.md](ASSUMPTIONS.md).

Run them with:

    python scripts/evaluate.py --field    --models ... --workers 8   # core conditions (Table 3 of the report)
    python scripts/evaluate.py --missions --models ... --workers 8   # the Phase 2 suite below

Flood is never trained on; it tests whether a policy carries over to a
disaster it has not seen.

## Core conditions (`--field`, Phase 1)

| Mission | What changes | Tests |
|---|---|---|
| `wild_mix` | nothing: lost person in the wilderness | the main wilderness result |
| `urban_mix` | nothing: survivors after a building collapse | urban search |
| `flood_mix_heldout` | nothing: people in and by flood water | carrying over to an unseen disaster |
| `wild_mix_adaptive_map` | the map estimates the real sensor rate | S1, M3 |
| `wild_hard_unresponsive` | the person never signals | P1 |
| `wild_hard_evasive` | the person hides from drones | P1 |
| `wild_hard_wrong_ipp` | the reported last-seen point is 0.6–1.5 km off | K1 |
| `wild_hard_outside` | the person is outside the area | K2 (nothing can be found) |
| `wild_hard_weak_sensor` | detection at 40% of the modelled rate | S1 |
| `wild_hard_late_start` | lost 12 hours before launch | P4, P8 |
| `wild_hard_radio` | 300 m radio range, drones return when out of contact | C1, D6 |
| `wild_hard_off_model` | the person follows none of the modelled behaviours | P3 |

## Phase 2 suite (`--missions`)

### Who is lost

| Mission | What changes | Tests |
|---|---|---|
| `wild_hiker` | hiker (distances 1.1/3.1/5.8/11.3 km) | P5, the commonest case |
| `wild_hunter` | hunter (follows drainages, responds) | P3, P5 |
| `wild_child` | child 7–12 (short distances, often hides) | P1, P5 |
| `wild_dementia` | person with dementia (rarely responds, short survival) | P1, P8 |
| `wild_despondent` | despondent person (close, or far at a chosen place; does not want to be found) | P1, P5 |
| `wild_group` | three people together | P6 |
| `wild_phone_ping` | a rough area from a phone ping | P2 |

### Conditions

| Mission | What changes | Tests |
|---|---|---|
| `wild_night` | launch at 02:00 | E4, S3 (thermal only) |
| `wild_cold_night` | 03:00, −8 °C, half hypothermic | S3, P8 (weak thermal contrast) |
| `wild_rain` | rain spells likely | E4, S3 |
| `wild_wind` | 11 m/s wind with gusts | D4, D6 |
| `wild_dense_forest` | poor terrain map, 2.5× false alarms | K5, S4 |

### Fleet and airspace

| Mission | What changes | Tests |
|---|---|---|
| `wild_two_drones` | two drones | team size |
| `wild_eight_drones` | eight drones | team size, coordination |
| `wild_short_battery` | 25-minute batteries, 5-minute swaps | D1 |
| `wild_airspace` | three no-fly zones and a temporary closure | D8 |

### Long operations and a growing area (K2)

Four-hour operations on a world built twice as wide. All three missions use
the same world, so the only difference is whether and when the area grows.

| Mission | What changes | Tests |
|---|---|---|
| `wild_long_fixed` | the area never grows | the reference |
| `wild_long_expand` | the area doubles in width once the map says the person is more likely outside | growth on evidence |
| `wild_long_expand_2h` | the area doubles at two hours | growth at an operational-period boundary |

### Urban collapse

| Mission | What changes | Tests |
|---|---|---|
| `urban_heavy_collapse` | 60% of buildings collapsed, unknown to the map | K5 |
| `urban_aftershock` | an aftershock mid-mission | E3 |
| `urban_smoke` | smoke from fires | E3, S3 |
| `urban_night` | launch at 02:00 | E4 |
| `urban_radio` | 200 m radio range, no relay | C1 |

### Flood (held out of training)

| Mission | What changes | Tests |
|---|---|---|
| `flood_rising_heldout` | water rises 2 m over the mission | E3 |
| `flood_night_heldout` | launch at 02:00 | E4 |

## Training missions

Training draws from the families in `FieldSampler` (wilderness and urban by
default). Every effect is randomised, so the suite's settings all appear
during training. Two options make the newer cases common enough to learn:

- `--expand-share`: the share of wilderness missions whose area can grow.
- `--long-share`: the share run as four-hour operations.

Both are off by default, which keeps Phase 1 runs reproducible. The Phase 2
settings ([PHASE2.md](PHASE2.md)) use 0.3 for each.
