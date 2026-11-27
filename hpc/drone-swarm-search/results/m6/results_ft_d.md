| Condition | Method | Mean find time (steps) | Find time (% of budget) | Time to 90% (steps) | Reached 90% | Found | Coverage | Repeats | Collisions | Shield stops | Alarms (false) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cam_zones_still | lawnmower | 51 ± 1 | 51% | 90 ± 15 | 43% | 81% | 99% | 5% | 0.0 | 0.8 | 20.8 (2.5) |
| cam_zones_still | belief_greedy | 43 ± 1 | 43% | 81 ± 25 | 48% | 83% | 95% | 11% | 0.0 | 2.3 | 36.7 (10.2) |
| cam_zones_still | mappo | 64 ± 1 | 64% | 98 ± 10 | 7% | 57% | 84% | 10% | 0.0 | 1.5 | 41.5 (9.4) |
| cam_zones_still | mappo_T0 | 59 ± 1 | 59% | 98 ± 8 | 7% | 63% | 87% | 11% | 0.0 | 1.7 | 36.5 (7.6) |
| cam_random_mix | lawnmower | 55 ± 2 | 55% | 89 ± 22 | 31% | 71% | 94% | 4% | 0.0 | 8.4 | 33.1 (4.6) |
| cam_random_mix | belief_greedy | 51 ± 2 | 51% | 85 ± 24 | 42% | 78% | 93% | 8% | 0.0 | 8.3 | 52.0 (15.3) |
| cam_random_mix | mappo | 66 ± 2 | 66% | 95 ± 16 | 12% | 58% | 90% | 7% | 0.0 | 3.6 | 50.7 (15.8) |
| cam_random_mix | mappo_T0 | 57 ± 2 | 57% | 90 ± 21 | 23% | 68% | 90% | 9% | 0.0 | 3.6 | 46.8 (14.1) |
