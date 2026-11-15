# Selected Videos — Fair 25-Video Subset

Curated subset of 25 full surgical videos copied from `../` (`full_surgical_videos/`, 110 videos total) for balanced development / GRPO experiments.

Each video directory contains a reconstructed full procedure (concatenated phase sub-clips) plus its timeline:

```text
selected_videos/
├── README.md
├── PH_0001_2931_S2/
│   ├── PH_0001_2931_S2.mp4
│   ├── PH_0001_2931_S2.timeline.json
│   └── PH_0001_2931_S2.timeline.txt
├── PH_0002_2932_S2/
└── ...
```

`timeline.json` fields per segment: `start_s`, `end_s`, `duration_s`, `order`, `phase`, `phase_name`, `occurrence`. Segments are contiguous and ordered by `order`. `timeline.txt` is the human-readable version (`start - end | phase | phase name`).

## Selection Criteria

1. **Duration near the mean:** full set mean `666.7 s` (11m07s), median `682.3 s`, range `375–901 s`. This subset is restricted to `581.8–731.2 s` (9.7–12.2 min), mean `663.0 s`, median `665.0 s`.
2. **Fair centers:** full set is `91× S1 (82.7%) / 19× S2 (17.3%)`. This subset is `20× S1 / 5× S2 (80%/20%)` — proportional, so S2 is represented without 3× over-sampling.
3. **All phases covered:** union of the 25 videos = `P01–P13` (all 13). No single S2 video contains all 13 phases, so 7× full-coverage (13-phase) S1 videos closest to the mean were included first, then filled with the next-closest videos to the overall mean.

No further re-encoding was done — files are copies of the parent folder.

## Inventory

| Property | Value |
|---|---:|
| Videos | 25 (20× S1, 5× S2) |
| Total duration | 4 h 36 m 15 s (16,575 s) |
| Mean video duration | 663.0 s (11m03s) |
| Median video duration | 665.0 s (11m05s) |
| Min / Max video | 581.8 s / 731.2 s |
| Segments (stitched sub-clips) | 819 |
| Mean segments / video | 32.8 |
| Disk usage | ~963 MB |
| Video codec | H.264, 480p, ~5 FPS nominal |
| Resolution | S1: 720×480, S2: 854×480 |
| Audio | AAC in 24/25; `PH_0018_2955_S2` has no audio stream |

## Video List (sorted by duration)

| Video | Site | Duration | Segments |
|---|---|---:|---:|
| `PH_0006_2936_S2` | S2 | 9m42s (581.8 s) | 26 |
| `PH_0018_2955_S2` | S2 | 10m10s (609.8 s) | 28 |
| `PH_0043_0096_S1` | S1 | 10m14s (614.0 s) | 34 |
| `PH_0012_2939_S2` | S2 | 10m17s (616.8 s) | 31 |
| `PH_0002_2932_S2` | S2 | 10m18s (618.2 s) | 24 |
| `PH_0071_0999_S1` | S1 | 10m39s (639.2 s) | 41 |
| `PH_0134_2178_S1` | S1 | 10m41s (641.4 s) | 29 |
| `PH_0111_1609_S1` | S1 | 10m46s (645.6 s) | 37 |
| `PH_0144_2448_S1` | S1 | 10m51s (650.8 s) | 29 |
| `PH_0078_1160_S1` | S1 | 10m58s (657.8 s) | 33 |
| `PH_0108_1529_S1` | S1 | 11m00s (660.2 s) | 32 |
| `PH_0146_2452_S1` | S1 | 11m04s (664.2 s) | 37 |
| `PH_0057_0239_S1` | S1 | 11m05s (665.0 s) | 39 |
| `PH_0116_1738_S1` | S1 | 11m07s (667.2 s) | 31 |
| `PH_0001_2931_S2` | S2 | 11m09s (669.2 s) | 27 |
| `PH_0136_2243_S1` | S1 | 11m17s (677.2 s) | 33 |
| `PH_0084_1203_S1` | S1 | 11m27s (687.4 s) | 31 |
| `PH_0064_0667_S1` | S1 | 11m30s (689.7 s) | 25 |
| `PH_0070_0882_S1` | S1 | 11m30s (689.8 s) | 26 |
| `PH_0086_1208_S1` | S1 | 11m34s (693.6 s) | 33 |
| `PH_0087_1221_S1` | S1 | 11m38s (698.4 s) | 40 |
| `PH_0081_1188_S1` | S1 | 11m40s (699.6 s) | 33 |
| `PH_0073_1074_S1` | S1 | 11m43s (703.0 s) | 33 |
| `PH_0077_1150_S1` | S1 | 11m44s (703.8 s) | 44 |
| `PH_0075_1093_S1` | S1 | 12m11s (731.2 s) | 43 |

Full-coverage (all 13 phases in one video): `PH_0043_0096_S1`, `PH_0111_1609_S1`, `PH_0144_2448_S1`, `PH_0086_1208_S1`, `PH_0087_1221_S1`, `PH_0077_1150_S1`, `PH_0075_1093_S1`.

## Phase Distribution (this subset)

| Phase | Segments | Videos covered (/25) | Total duration | Mean segment |
|---|---:|---:|---:|---:|
| `P01` Incision | 55 | 25 | 15m12s (912.5 s) | 16.6 s |
| `P02` Viscoelastic | 77 | 25 | 16m49s (1009.2 s) | 13.1 s |
| `P03` Capsulorhexis | 31 | 25 | 23m24s (1404.1 s) | 45.3 s |
| `P04` Hydrodissection | 25 | 25 | 8m47s (527.4 s) | 21.1 s |
| `P05` Phacoemulsification | 26 | 25 | 53m27s (3207.3 s) | 123.4 s |
| `P06` Irrigation-Aspiration | 41 | 24 | 24m25s (1465.0 s) | 35.7 s |
| `P07` Capsule Polishing | 22 | 20 | 4m11s (251.4 s) | 11.4 s |
| `P08` Lens Implantation | 26 | 25 | 6m54s (413.8 s) | 15.9 s |
| `P09` Lens Positioning | 24 | 22 | 7m20s (440.1 s) | 18.3 s |
| `P10` Viscoelastic Suction | 32 | 25 | 16m43s (1003.1 s) | 31.3 s |
| `P11` Anterior Chamber Flushing | 12 | 10 | 2m07s (126.7 s) | 10.6 s |
| `P12` Tonifying-Antibiotics | 32 | 23 | 9m08s (548.5 s) | 17.1 s |
| `P13` Idle | 416 | 25 | 1h27m05s (5225.1 s) | 12.6 s |

## Surgical Phases

| ID | Phase |
|---|---|
| `P01` | Incision |
| `P02` | Viscoelastic |
| `P03` | Capsulorhexis |
| `P04` | Hydrodissection |
| `P05` | Phacoemulsification |
| `P06` | Irrigation-Aspiration |
| `P07` | Capsule Polishing |
| `P08` | Lens Implantation |
| `P09` | Lens Positioning |
| `P10` | Viscoelastic Suction |
| `P11` | Anterior Chamber Flushing |
| `P12` | Tonifying-Antibiotics |
| `P13` | Idle |

## Caveats

- `P13` Idle is a real class (~31.5% of subset time, ~50.8% of segments). Do not drop it silently.
- `P11` is rare by design (only 21/110 videos in the parent set contain it; 10/25 here). Expect high variance on `P11` metrics.
- `P07` (20/25) and `P09` (22/25) are also incomplete by video — use segment- or time-weighted metrics.
- Phases repeat within a video (Idle interleaves; `P01/P02/P06` commonly recur). Do not assume strict `P01→P12` order.
- Filenames `PH_<ClipID>_<RawVideoID>_S<Site>` encode site (`S1`/`S2`); resolution follows site.
- `avg_frame_rate` metadata jitters around 5 FPS due to concatenation; treat as 5 FPS.
