# Stage 2 vs Stage 1 (Same Protocol)

Protocol:
- Test set: `hgr-stage1-large/test_set.json`
- Generation: `num_beams=4`, `max_length=48`
- Metrics: SacreBLEU + chrF++

## Results

| Model | Overall BLEU | Overall chrF++ | mhi BLEU | him BLEU |
|---|---:|---:|---:|---:|
| Stage 1 Large | 22.4497 | 47.6631 | 22.8772 | 21.8421 |
| Stage 2 DPO (full cleaned-data run) | 21.8603 | 46.8517 | 23.5066 | 19.8539 |

## Delta (Stage2 - Stage1)

- Overall BLEU: **-0.5894**
- Overall chrF++: **-0.8114**
- mhi BLEU: **+0.6294**
- him BLEU: **-1.9882**

## Interpretation

Stage 2 DPO is not universally worse; it improved one direction (`mhi`) but substantially regressed the reverse direction (`him`), yielding net degradation.
