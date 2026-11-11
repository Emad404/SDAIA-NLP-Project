# Bayan: serving benchmarks (auto-generated)

## Optimisation ladder: `topic_xlmr_l4_kd`

Hardware: {'cpu_model': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'logical_cores': 2, 'python': '3.13.16', 'torch': '2.14.1+cu130', 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39'}

| Rung | val macro-F1 | accuracy tax vs R1 | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| R0_torch_fp32_pad128 | 0.8754 | +0.0 | 88.89 | 250.95 | 388.05 |
| R1_torch_fp32_dynamic_pad | 0.8754 | +0.0 | 38.14 | 115.43 | 270.55 |
| R2_ort_fp32 | 0.8754 | +0.0 | 20.19 | 52.04 | 63.06 |
| R3_ort_int8_dynamic | 0.8765 | +0.0011 | 11.64 | 20.3 | 27.63 |

Sizes (MB): {'onnx_fp32': 885.5, 'onnx_int8': 221.6}  |  parity: {'R2_vs_R1_argmax_agreement': 1.0, 'R3_vs_R1_argmax_agreement': 0.996328029375765, 'R2_max_abs_prob_diff': 4.231929779052734e-06, 'R3_max_abs_prob_diff': 0.5643066167831421}

Notes: batch=1, warm-up 30 requests discarded, 300 real val messages x2 passes, threads pinned, no concurrency (HTTP/16-user test is s07)

## Optimisation ladder: `topic_xlmr_l3_kd`

Hardware: {'cpu_model': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'logical_cores': 2, 'python': '3.13.16', 'torch': '2.14.1+cu130', 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39'}

| Rung | val macro-F1 | accuracy tax vs R1 | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| R0_torch_fp32_pad128 | 0.8778 | +0.0 | 66.13 | 201.57 | 302.1 |
| R1_torch_fp32_dynamic_pad | 0.8778 | +0.0 | 29.52 | 61.95 | 205.08 |
| R2_ort_fp32 | 0.8778 | +0.0 | 15.01 | 31.68 | 41.6 |
| R3_ort_int8_dynamic | 0.8776 | -0.0002 | 10.47 | 31.25 | 39.69 |

Sizes (MB): {'onnx_fp32': 857.1, 'onnx_int8': 214.5}  |  parity: {'R2_vs_R1_argmax_agreement': 1.0, 'R3_vs_R1_argmax_agreement': 0.9908200734394125, 'R2_max_abs_prob_diff': 3.844499588012695e-06, 'R3_max_abs_prob_diff': 0.19655126333236694}

Notes: batch=1, warm-up 30 requests discarded, 300 real val messages x2 passes, threads pinned, no concurrency (HTTP/16-user test is s07)

## Optimisation ladder: `topic_xlmr_all_ca`

Hardware: {'cpu_model': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'logical_cores': 2, 'python': '3.13.16', 'torch': '2.14.1+cu130', 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39'}

| Rung | val macro-F1 | accuracy tax vs R1 | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| R0_torch_fp32_pad128 | 0.8817 | +0.0 | 273.9 | 793.33 | 990.26 |
| R1_torch_fp32_dynamic_pad | 0.8817 | +0.0 | 113.8 | 344.37 | 623.4 |
| R2_ort_fp32 | 0.8817 | +0.0 | 58.99 | 149.66 | 179.98 |
| R3_ort_int8_dynamic | 0.8766 | -0.0051 | 37.3 | 106.92 | 141.55 |

Sizes (MB): {'onnx_fp32': 1112.4, 'onnx_int8': 278.8}  |  parity: {'R2_vs_R1_argmax_agreement': 1.0, 'R3_vs_R1_argmax_agreement': 0.9810281517747858, 'R2_max_abs_prob_diff': 9.953975677490234e-06, 'R3_max_abs_prob_diff': 0.9007695913314819}

Notes: batch=1, warm-up 30 requests discarded, 300 real val messages x2 passes, threads pinned, no concurrency (HTTP/16-user test is s07)

## HTTP benchmark: shipped configuration (POST /v1/classify, 16 concurrent users)

Server pinned to CPU 0 and the load generator to CPU 1, 1 ONNX thread, no micro-batch wait. `int8` is the default precision, `fp32` is the rollback. 'Think time' is the mean exponential pause each simulated user takes between requests (0 = closed loop at saturation).

| run | think time / user | requests | errors | rps | p50 | p95 | p99 | max | p99<=40ms |
|---|---|---|---|---|---|---|---|---|---|
| int8final_int8 | none (saturation) | 1976 | 0 | 89.8 | 172.29 | 226.4 | 248.65 | 274.38 | False |
| int8final_int8 | 250 ms | 1323 | 0 | 60.1 | 21.84 | 51.99 | 68.02 | 77.86 | False |
| int8final_int8 | 500 ms | 725 | 0 | 33.0 | 14.51 | 32.63 | 43.85 | 50.6 | False |
| int8final_int8 | 1000 ms | 359 | 0 | 16.3 | 12.19 | 25.06 | 33.98 | 43.04 | True |
| fp32final_fp32 | 500 ms | 704 | 0 | 32.0 | 26.06 | 69.18 | 95.16 | 119.59 | False |
| fp32final_fp32 | 1000 ms | 352 | 0 | 16.0 | 19.99 | 47.65 | 67.91 | 122.96 | False |

## HTTP benchmark: tuning history (not the shipped configuration)

Earlier runs used 2 ONNX threads, a 2 ms batching wait and an unpinned load generator; they explain why the final settings were chosen. Saturation numbers of unpinned runs are contaminated by the load generator competing for the same 2 vCPUs.

| run | think time / user | requests | errors | rps | p50 | p95 | p99 | max | p99<=40ms |
|---|---|---|---|---|---|---|---|---|---|
| int8_topic_xlmr_l4_kd | none (saturation) | 679 | 0 | 39.9 | 353.36 | 649.21 | 699.59 | 816.16 | False |
| int8_topic_xlmr_l4_kd | 100 ms | 760 | 0 | 44.7 | 230.73 | 519.94 | 610.21 | 653.2 | False |
| int8_topic_xlmr_l4_kd | 250 ms | 754 | 0 | 44.4 | 91.34 | 384.42 | 468.29 | 501.13 | False |
| int8_topic_xlmr_l4_kd | 500 ms | 499 | 0 | 29.4 | 35.85 | 201.03 | 277.66 | 305.84 | False |
| int8pin | none (saturation) | 1464 | 0 | 86.1 | 177.82 | 239.62 | 260.96 | 279.49 | False |
| int8pin | 250 ms | 1032 | 0 | 60.7 | 20.71 | 60.7 | 81.22 | 104.74 | False |
| int8pin | 500 ms | 543 | 0 | 31.9 | 13.78 | 35.75 | 46.07 | 63.78 | False |
| int8pin | 1000 ms | 268 | 0 | 15.8 | 12.35 | 23.26 | 31.2 | 48.69 | True |
| fp32_topic_xlmr_all_ca | none (saturation) | 193 | 0 | 11.4 | 1298.83 | 2248.31 | 2267.94 | 2281.65 | False |
| fp32_topic_xlmr_all_ca | 250 ms | 218 | 0 | 12.8 | 846.42 | 2162.07 | 2635.8 | 2729.34 | False |
| int8_topic_xlmr_l3_kd | none (saturation) | 953 | 0 | 56.1 | 253.12 | 497.21 | 590.46 | 633.03 | False |
| int8_topic_xlmr_l3_kd | 100 ms | 1002 | 0 | 58.9 | 144.41 | 391.17 | 558.63 | 731.29 | False |
| int8_topic_xlmr_l3_kd | 250 ms | 859 | 0 | 50.5 | 49.5 | 243.12 | 333.68 | 454.64 | False |
| int8_topic_xlmr_l3_kd | 500 ms | 516 | 0 | 30.4 | 22.87 | 113.28 | 147.79 | 179.74 | False |
| fp32_topic_xlmr_l4_kd | none (saturation) | 528 | 0 | 31.1 | 463.32 | 796.05 | 814.14 | 823.46 | False |
| fp32_topic_xlmr_l4_kd | 250 ms | 575 | 0 | 33.8 | 178.41 | 521.97 | 605.46 | 716.73 | False |
| int8_topic_xlmr_all_ca | none (saturation) | 257 | 0 | 15.1 | 912.72 | 1989.72 | 2006.25 | 2030.24 | False |
| int8_topic_xlmr_all_ca | 100 ms | 278 | 0 | 16.4 | 817.32 | 1439.49 | 1735.72 | 1767.41 | False |
| int8_topic_xlmr_all_ca | 250 ms | 280 | 0 | 16.5 | 653.11 | 1491.27 | 1617.0 | 1668.69 | False |
| int8_topic_xlmr_all_ca | 500 ms | 294 | 0 | 17.3 | 278.76 | 1264.06 | 1718.65 | 1787.97 | False |
| int8t1w0 | 250 ms | 1022 | 0 | 60.1 | 20.22 | 75.48 | 100.29 | 134.65 | False |
| int8t1w0 | 500 ms | 541 | 0 | 31.8 | 13.68 | 35.07 | 51.56 | 75.18 | False |
| int8t1w0 | 1000 ms | 271 | 0 | 15.9 | 12.07 | 37.64 | 55.01 | 79.31 | False |

Caveat: load generator and server share the same 2-vCPU Colab VM; the lab CPU may differ
Hardware: {'cpu_model': 'Intel(R) Xeon(R) CPU @ 2.00GHz', 'logical_cores': 2, 'python': '3.13.16', 'torch': '2.14.1+cu130', 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39'}
