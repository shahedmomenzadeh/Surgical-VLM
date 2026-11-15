# Evaluation dashboard

## Tier 0 — deterministic (lexical vs teacher + protocol reliability; ↑ = higher better, ↓ = lower better)

| source | videos | ROUGE-L ↑ | instrF1 ↑ | echo ↓ | empty ↓ | nonIdle-RougeL ↑ |
|---|---|---|---|---|---|---|
| hulumed-4b.s1 | 25 | 0.117 | 0.176 | 0.0 | 0.0 | 0.107 |
| hulumed-4b.s3 | 25 | 0.111 | 0.542 | 0.0 | 0.0 | 0.094 |
| hulumed-7b.s1 | 25 | 0.096 | 0.254 | 0.0 | 0.0 | 0.08 |
| hulumed-7b.s3 | 25 | 0.111 | 0.745 | 0.0 | 0.0 | 0.101 |
| lingshu-7b.s1 | 25 | 0.149 | 0.125 | 0.0 | 0.0 | 0.143 |
| lingshu-7b.s3 | 25 | 0.141 | 0.439 | 0.0 | 0.0 | 0.14 |
| qwen3vl-4b-instruct.s1 | 25 | 0.184 | 0.146 | 0.0 | 0.0 | 0.178 |
| qwen3vl-4b-instruct.s3 | 25 | 0.239 | 0.611 | 0.0 | 0.0 | 0.253 |
| qwen3vl-2b-instruct.s1 | 25 | 0.194 | 0.088 | 0.0 | 0.0 | 0.2 |
| qwen3vl-2b-instruct.s3 | 25 | 0.152 | 0.645 | 0.0 | 0.0 | 0.151 |
| qwen3vl-2b-sft-stage2.s1 | 25 | 0.131 | 0.148 | 0.0 | 0.0 | 0.111 |
| qwen3vl-2b-sft-stage2.s3 | 25 | 0.19 | 0.624 | 0.0 | 0.0 | 0.2 |
| qwen3vl-2b-grpo.s1 | 25 | 0.15 | 0.16 | 0.0 | 0.0 | 0.14 |
| qwen3vl-2b-grpo.s3 | 25 | 0.181 | 0.546 | 0.0 | 0.0 | 0.188 |

## Tier J — video-grounded judge (mean 0-3 per video; 3 = best, 0 = worst)

| source | videos | judged | accuracy ↑ | coverage ↑ | nonIdle-acc ↑ | nonIdle-cov ↑ | material ↓ | unassess |
|---|---|---|---|---|---|---|---|---|
| hulumed-4b.s1 | 25 | 1.0 | 1.018 | 1.084 | 0.841 | 0.833 | 0.857 | 0.003 |
| hulumed-4b.s3 | 25 | 1.0 | 1.616 | 1.553 | 1.272 | 1.128 | 0.485 | 0.013 |
| hulumed-7b.s1 | 25 | 1.0 | 1.072 | 1.121 | 0.901 | 0.866 | 0.833 | 0.003 |
| hulumed-7b.s3 | 25 | 1.0 | 1.555 | 1.557 | 1.549 | 1.422 | 0.522 | 0.01 |
| lingshu-7b.s1 | 25 | 1.0 | 0.986 | 1.021 | 0.917 | 0.923 | 0.884 | 0.004 |
| lingshu-7b.s3 | 25 | 1.0 | 1.331 | 1.4 | 1.19 | 1.19 | 0.65 | 0.011 |
| qwen3vl-4b-instruct.s1 | 25 | 1.0 | 1.097 | 1.143 | 0.888 | 0.855 | 0.759 | 0.013 |
| qwen3vl-4b-instruct.s3 | 25 | 1.0 | 1.205 | 1.351 | 1.233 | 1.361 | 0.763 | 0.009 |
| qwen3vl-2b-instruct.s1 | 25 | 1.0 | 0.919 | 1.02 | 0.873 | 0.909 | 0.939 | 0.006 |
| qwen3vl-2b-instruct.s3 | 25 | 1.0 | 1.087 | 1.214 | 1.125 | 1.28 | 0.889 | 0.002 |
| qwen3vl-2b-sft-stage2.s1 | 25 | 1.0 | 1.118 | 1.151 | 0.789 | 0.739 | 0.729 | 0.011 |
| qwen3vl-2b-sft-stage2.s3 | 25 | 1.0 | 1.308 | 1.445 | 1.271 | 1.325 | 0.698 | 0.009 |
| qwen3vl-2b-grpo.s1 | 25 | 1.0 | 1.13 | 1.185 | 0.976 | 0.974 | 0.798 | 0.012 |
| qwen3vl-2b-grpo.s3 | 25 | 1.0 | 1.625 | 1.708 | 1.484 | 1.448 | 0.436 | 0.033 |

Notes: Tier J accuracy/coverage are QUALITY grades — higher is better (0 = no usable description, 1 = material error, 2 = minor error, 3 = correct); they are judged independently (a vague but harmless report scores well on accuracy and poorly on coverage). material = share of chunks with a material issue (lower is better); unassess = share with an abstained detail (descriptive, not scored).

## Setting contrasts (s3−s1, paired per-video, chunk-matched, 95% bootstrap CI)

```json
{
 "deterministic": {
  "hulumed-4b": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": -0.0064,
    "ci95": [
     -0.0203,
     0.007
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.3661,
    "ci95": [
     0.3112,
     0.4187
    ]
   }
  },
  "hulumed-7b": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": 0.0153,
    "ci95": [
     0.0068,
     0.0233
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.4915,
    "ci95": [
     0.4273,
     0.551
    ]
   }
  },
  "lingshu-7b": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": -0.0078,
    "ci95": [
     -0.0219,
     0.0075
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.3137,
    "ci95": [
     0.2784,
     0.3518
    ]
   }
  },
  "qwen3vl-4b-instruct": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": 0.0554,
    "ci95": [
     0.0389,
     0.0729
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.4653,
    "ci95": [
     0.4126,
     0.5175
    ]
   }
  },
  "qwen3vl-2b-instruct": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": -0.042,
    "ci95": [
     -0.0523,
     -0.0307
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.5567,
    "ci95": [
     0.525,
     0.5915
    ]
   }
  },
  "qwen3vl-2b-sft-stage2": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": 0.0588,
    "ci95": [
     0.0451,
     0.071
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.4761,
    "ci95": [
     0.4269,
     0.5218
    ]
   }
  },
  "qwen3vl-2b-grpo": {
   "s3-s1_rougeL": {
    "n_videos": 25,
    "mean_delta": 0.0313,
    "ci95": [
     0.0243,
     0.0382
    ]
   },
   "s3-s1_instrF1": {
    "n_videos": 25,
    "mean_delta": 0.3857,
    "ci95": [
     0.3412,
     0.4286
    ]
   }
  }
 },
 "judge": {
  "hulumed-4b": {
   "s3-s1_accuracy": {
    "n_videos": 25,
    "mean_delta": 0.5978,
    "ci95": [
     0.4993,
     0.6986
    ]
   },
   "s3-s1_coverage": {
    "n_videos": 25,
    "mean_delta": 0.4691,
    "ci95": [
     0.3915,
     0.5437
    ]
   }
  },
  "hulumed-7b": {
   "s3-s1_accuracy": {
    "n_videos": 25,
    "mean_delta": 0.4826,
    "ci95": [
     0.3626,
     0.5965
    ]
   },
   "s3-s1_coverage": {
    "n_videos": 25,
    "mean_delta": 0.4355,
    "ci95": [
     0.34,
     0.5265
    ]
   }
  },
  "lingshu-7b": {
   "s3-s1_accuracy": {
    "n_videos": 25,
    "mean_delta": 0.3454,
    "ci95": [
     0.2481,
     0.4397
    ]
   },
   "s3-s1_coverage": {
    "n_videos": 25,
    "mean_delta": 0.3789,
    "ci95": [
     0.2958,
     0.4694
    ]
   }
  },
  "qwen3vl-4b-instruct": {
   "s3-s1_accuracy": {
    "n_videos": 25,
    "mean_delta": 0.1081,
    "ci95": [
     0.0406,
     0.1676
    ]
   },
   "s3-s1_coverage": {
    "n_videos": 25,
    "mean_delta": 0.2074,
    "ci95": [
     0.118,
     0.2902
    ]
   }
  },
  "qwen3vl-2b-instruct": {
   "s3-s1_accuracy": {
    "n_videos"
```

## s1 phase recognition (deterministic, code)

```json
{
 "hulumed-4b": {
  "videos": 25,
  "mean_accuracy": 0.061
 },
 "hulumed-7b": {
  "videos": 25,
  "mean_accuracy": 0.051
 },
 "lingshu-7b": {
  "videos": 25,
  "mean_accuracy": 0.045
 },
 "qwen3vl-4b-instruct": {
  "videos": 25,
  "mean_accuracy": 0.119
 },
 "qwen3vl-2b-instruct": {
  "videos": 25,
  "mean_accuracy": 0.069
 },
 "qwen3vl-2b-sft-stage2": {
  "videos": 25,
  "mean_accuracy": 0.169
 },
 "qwen3vl-2b-grpo": {
  "videos": 25,
  "mean_accuracy": 0.051
 }
}
```
