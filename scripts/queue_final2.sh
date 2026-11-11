#!/bin/bash
# finalise config -> quantisation decisions -> final HTTP bench (pinned cores) -> error analysis -> unit tests
source /content/env.sh
cd /content/bayan_project
L=$BAYAN_WORK/logs
python scripts/s07e_finalize_config.py > $L/f2_s07e.log 2>&1
python scripts/s06c_quant_decisions.py > $L/f2_s06c.log 2>&1
python scripts/s07c_http_bench.py --precision int8 --pin --max-wait-ms 0 --threads 1 --seconds 25 --think-ms-list 0,250,500,1000 --tag final_int8 > $L/f2_bench_int8.log 2>&1
python scripts/s07c_http_bench.py --precision fp32 --pin --max-wait-ms 0 --threads 1 --seconds 25 --think-ms-list 500,1000 --tag final_fp32 > $L/f2_bench_fp32.log 2>&1
python scripts/s10_error_analysis.py > $L/f2_s10.log 2>&1
python -m pytest -q tests > $L/f2_pytest.log 2>&1
echo DONE > $L/f2_done.txt
