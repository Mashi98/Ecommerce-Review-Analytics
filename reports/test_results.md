# Final test-set evaluation

Both models scored once on the held-out test split (3,386 reviews). DistilBERT predictions come from the final Colab run (weights saved to Drive); the baseline is retrained locally with the reported settings and checked to reproduce its validation macro-F1 of 0.6245 first.

## Summary

| model      | split      | accuracy | macro_f1 | weighted_f1 |
|------------|------------|----------|----------|-------------|
| baseline   | validation | 0.7951   | 0.6245   | 0.8083      |
| baseline   | test       | 0.7930   | 0.6182   | 0.8059      |
| distilbert | validation | 0.8288   | 0.6785   | 0.8411      |
| distilbert | test       | 0.8269   | 0.6670   | 0.8379      |

Test macro-F1: baseline 0.6182, DistilBERT 0.6670.

## Is the gain real?

Paired bootstrap over test reviews (2,000 resamples): DistilBERT minus baseline macro-F1 = **+0.0488**, 95% interval [+0.0249, +0.0741]. DistilBERT failed to beat the baseline in 0.0% of resamples.

## Per class (test)

| class    | baseline P / R / F1   | distilbert P / R / F1 | support |
|----------|-----------------------|-----------------------|---------|
| negative | 0.491 / 0.598 / 0.539 | 0.585 / 0.668 / 0.624 | 356     |
| neutral  | 0.364 / 0.467 / 0.409 | 0.397 / 0.512 / 0.447 | 420     |
| positive | 0.943 / 0.872 / 0.906 | 0.963 / 0.899 / 0.930 | 2610    |

## Confusion matrix (test): baseline

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 213      | 98      | 45       |
| neutral  | 131      | 196     | 93       |
| positive | 90       | 244     | 2276     |

## Confusion matrix (test): distilbert

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 238      | 98      | 20       |
| neutral  | 135      | 215     | 70       |
| positive | 34       | 229     | 2347     |
