# Scoring Notes

Code hien tai da cham phan online ERS trong `benchmark/metrics.py` va `scripts/calculate_ers.py`.

## Online ERS

Moi request thanh cong:

```text
x_ttft = clamp((400 - TTFT) / (400 - 10), 0, 1)
s_ttft = x_ttft ** 2
x_tpot = clamp((10 - TPOT_mean) / (10 - 1), 0, 1)
s_tpot = x_tpot ** 2
request_score = 0.5 * s_ttft + 0.5 * s_tpot
```

Error, timeout, hoac response co `output_token_count = 0`:

```text
request_score = 0
```

Final online ERS:

```text
ERS = mean(request_score for all requests)
```

TPOT trong benchmark duoc tinh theo:

```text
TPOT = (last_token_time - first_token_time) / max(output_token_count - 1, 1)
```

Khong tinh TPOT bang total request latency chia output token.

## Accuracy Gate

Sau vong online, BTC co the chay GPQA cho toi da 5 submissions tot nhat. Goi:

```text
delta = baseline_accuracy - submission_accuracy
```

Penalty:

```text
f(delta) = 1.0                         neu delta <= 0.10
f(delta) = 1.0 - (delta - 0.10) / 0.06  neu 0.10 < delta < 0.16
f(delta) = 0.0                         neu delta >= 0.16
```

Diem cuoi:

```text
final_score = 100 * ERS * f(delta)
```

## Commands

Tinh ERS tu request results:

```bash
python scripts/calculate_ers.py results/<experiment_id>/requests.json
```

Tinh diem cuoi khi da co accuracy:

```bash
python scripts/calculate_final_score.py \
  --aggregate-json results/<experiment_id>/aggregate.json \
  --baseline-accuracy 0.40 \
  --submission-accuracy 0.38
```

Hoac nhap ERS truc tiep:

```bash
python scripts/calculate_final_score.py \
  --ers 0.72 \
  --baseline-accuracy 0.40 \
  --submission-accuracy 0.38
```

