# Relabel request to Ahmed Berat Özer (ready to send)

Send as a Kaggle direct message to **ahmedberatozer**, or as a comment on the dataset page
https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models (Discussion tab).

---

**Subject:** Licence of your CASMI26 datasets — could they be MIT / CC BY 4.0?

Hi Ahmed,

thank you for publishing your CASMI 2026 pipeline (casmi26-v4b-models and the related datasets). It's one of the most
useful public resources in this competition.

A question about the licence. The dataset descriptions say *"non-commercial, with attribution"*, because the Rules
page lists the Competition Data as CC BY-NC 4.0. Since then the host has clarified in
https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743236 that models trained on
train.parquet may be open-sourced. The prize rules require winning solutions to be releasable under an open licence
(MIT), so anything labelled non-commercial can't be part of a prize-eligible solution, for us or for anyone else.

Would you consider relabelling these datasets under MIT, Apache-2.0 or CC BY 4.0, keeping the attribution?

- ahmedberatozer/casmi26-v4b-models
- ahmedberatozer/casmi26-v3-models
- ahmedberatozer/casmi26-v2-pool
- ahmedberatozer/casmi26-fpnet-full1

(The DreaMS-derived fe_D.pt already notes MIT / CC BY 4.0 for the base weights.) If you'd rather keep them
non-commercial, that's completely fine. We just wanted to ask before re-implementing anything ourselves.

Thanks, and good luck in the competition!

---

Why this matters (internal): if he relabels, the 0.384–0.399-class public pipeline becomes eligible input (see
`research/analysis/fork_licence_rebuild_plan.md` §2, §5). If he declines or doesn't answer, nothing changes: we
re-implement ideas only, never his code, weights or tables.
