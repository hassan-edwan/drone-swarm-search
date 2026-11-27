| Condition | Method | Mean find time (steps) | Find time (% of budget) | Time to 90% (steps) | Reached 90% | Found | Coverage | Repeats | Collisions | Shield stops | Alarms (false) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam_zones_still | lawnmower | 51 ± 1 | 51% | 90 ± 15 | 43% | 81% | 99% | 5% | 0.0 | 0.8 | 20.8 (2.5) |
| cam_zones_still | belief_greedy | 43 ± 1 | 43% | 81 ± 25 | 48% | 83% | 95% | 11% | 0.0 | 2.3 | 36.7 (10.2) |
| cam_zones_still | mappo | 55 ± 1 | 55% | 95 ± 14 | 15% | 67% | 84% | 8% | 0.0 | 1.6 | 30.8 (6.1) |
| cam_zones_still | mappo_T0 | 58 ± 1 | 58% | 96 ± 13 | 9% | 61% | 81% | 8% | 0.0 | 1.2 | 29.1 (5.6) |
| cam_random_mix | lawnmower | 55 ± 2 | 55% | 89 ± 22 | 31% | 71% | 94% | 4% | 0.0 | 8.4 | 33.1 (4.6) |
| cam_random_mix | belief_greedy | 51 ± 2 | 51% | 85 ± 24 | 42% | 78% | 93% | 8% | 0.0 | 8.3 | 52.0 (15.3) |
| cam_random_mix | mappo | 59 ± 2 | 59% | 94 ± 18 | 14% | 61% | 85% | 7% | 0.0 | 3.9 | 34.7 (9.3) |
| cam_random_mix | mappo_T0 | 62 ± 2 | 62% | 95 ± 17 | 10% | 55% | 84% | 9% | 0.0 | 4.0 | 33.1 (9.4) |
