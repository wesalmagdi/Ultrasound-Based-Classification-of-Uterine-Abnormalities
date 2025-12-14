Tabular branch

1. Normalize features

2. Feed into Dense layers

3. Extract abstract features

-> Tabular features → Dense(64) → Dense(32)

Image branch

1. CNN extracts spatial patterns

-> Image → Conv → Pool → Conv → Flatten

Fusion

`. Combine both representations

-> Concatenate([image_features, tabular_features])

----------------
⚠️ One thing you must NOT do

❌ Do NOT evaluate your model without reporting:

CNN-only baseline

CNN + radiomics baseline

Otherwise reviewers may say:

“Your performance comes mainly from metadata.”