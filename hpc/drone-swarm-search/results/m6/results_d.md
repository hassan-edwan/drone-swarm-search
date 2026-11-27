| Condition | Method | Mean find time (steps) | Find time (% of budget) | Time to 90% (steps) | Reached 90% | Found | Coverage | Repeats | Collisions | Shield stops | Alarms (false) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam_zones_still | lawnmower | 51 ± 1 | 51% | 90 ± 15 | 43% | 81% | 99% | 5% | 0.0 | 0.8 | 20.8 (2.5) |
| cam_zones_still | belief_greedy | 43 ± 1 | 43% | 81 ± 25 | 48% | 83% | 95% | 11% | 0.0 | 2.3 | 36.7 (10.2) |
| cam_zones_still | mappo | 54 ± 1 | 54% | 94 ± 15 | 18% | 71% | 49% | 40% | 0.0 | 2.8 | 0.1 (0.0) |
| cam_zones_still | mappo_T0 | 54 ± 1 | 54% | 95 ± 14 | 16% | 70% | 49% | 39% | 0.0 | 3.0 | 0.1 (0.0) |
| cam_random_mix | lawnmower | 55 ± 2 | 55% | 89 ± 22 | 31% | 71% | 94% | 4% | 0.0 | 8.4 | 33.1 (4.6) |
| cam_random_mix | belief_greedy | 51 ± 2 | 51% | 85 ± 24 | 42% | 78% | 93% | 8% | 0.0 | 8.3 | 52.0 (15.3) |
| cam_random_mix | mappo | 59 ± 2 | 59% | 91 ± 21 | 25% | 64% | 62% | 14% | 0.0 | 5.8 | 0.5 (0.1) |
| cam_random_mix | mappo_T0 | 58 ± 2 | 58% | 91 ± 22 | 20% | 65% | 63% | 12% | 0.0 | 6.5 | 0.5 (0.1) |
