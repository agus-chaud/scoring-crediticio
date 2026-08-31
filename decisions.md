# Technical decisions

## DEC-001 — Reconstructed production preprocessing and three risk models

- **Area:** deployment
- **Decision:** Rebuild the raw-input preprocessing inside each released sklearn pipeline and train three independent models: PD, EAD and LGD.
- **Alternative discarded:** Reuse the already transformed tables as the public API contract.
- **Why discarded:** Those tables contain engineered numeric columns only; making them the API input would expose an unusable contract and would not reproduce the raw-loan preparation.
- **Conclusion:** The API receives 14 raw loan fields. Each model owns the same deterministic preparation chain, fitted only on training data. EAD and LGD are trained on defaulted loans, as defined by the original notebooks.

## DEC-002 — Production-safe rebuild differs from the historical notebook

- **Area:** deployment
- **Decision:** Use `random_state=42`, fit preparation inside model pipelines, and apply `OneHotEncoder(drop='first')`.
- **Alternative discarded:** Copy the historical notebook behavior verbatim.
- **Why discarded:** The notebook did not fix its split seed and fitted encoders/scalers before validation, which can make validation look better than it really is.
- **Conclusion:** The released artifact favors repeatable, safer training over byte-for-byte reproduction of the historic prepared tables.
