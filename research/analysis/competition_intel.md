# Competition intelligence digest: CASMI 2026 (Kaggle `enveda-CASMI26-molecule-id-mass-spectra`)

Task E (Hermes). Compiled 2026-10-01. Read-only: no notebook was forked, copied, run or submitted.

**Method.** Everything below was read through the Kaggle MCP API (`search_notebooks`, `get_notebook_info`, `get_dataset_info`, `list_competition_topics`, `get_forum_topic` with all comments, `list_competition_pages`, `get_competition_leaderboard`).

**Coverage.**
- About 100 notebooks:
  - the 27 most-voted;
  - pages 1–2 of Kaggle's "score descending" sort (75 notebooks);
  - the full ahmedberatozer v4 series;
  - three data-analysis notebooks.
- All 58 discussion topics. The listing count was 58 on 2026-10-01.
- All 11 competition pages.

The per-batch raw notes are kept in the session scratchpad and are not part of the repo.

**Labels.**
- **FACT** means a quote or number read directly from the linked source. Numbers are quoted exactly as written. "LB" means the public leaderboard.
- **CV** marks a local or proxy number reported by the author. It is not an LB score.
- **INFERENCE** marks my own reading.
- **CLAIM** marks a number in a title or header that the notebook does not back up (targets, "0.380+" and similar).

**Caveats on sources.**
- `get_notebook_info` reports `last_run_time` = 2026-09-27 18:39–18:54 UTC for every notebook. INFERENCE: this is a platform-wide re-run stamp. The dates given below are the `search_notebooks` listing dates, which track the latest version better. Where a notebook states its own date, that date is used.
- The API returns no notebook-level licence. Kaggle public notebooks default to Apache 2.0 (INFERENCE), unless the notebook says otherwise.
- Kaggle's per-notebook "score" is not exposed through the API. A "stated LB" is therefore always text written inside the notebook.
- Two agents saw a tool-result file collision: `get_notebook_info` returned a different notebook. Each affected fetch was repeated and the notebook ref was checked before parsing.

Links: notebooks are `https://www.kaggle.com/code/<ref>`, and discussions are `https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>` (written as **D/<id>**).

---

## 0. Executive summary

1. **The leaderboard snapshot on 2026-10-01 (FACT).**
   - Top 5: pikachu 0.464, Randy 0.435, Ozymandias31415 0.425, Shehab Alshehabi 0.423, chopper 0.421. Next: Udam Liyanage 0.414, Komil Parmar 0.409.
   - Ranks 15–40 are all **0.399**, submitted on 09-30. INFERENCE: the public 0.399 notebook is being forked widely.
   - 1,917 teams. The deadline is 2026-12-14.
2. **The best public pipeline is 0.399** (FACT: [seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399](https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399), checked against our local copy in `external/public_nb/v4n399/`).
   - It is Ahmed Berat Özer's v4n stack plus a second engine, fused by reciprocal rank. ICEBERG and GLACIER then re-score the fused union inside same-formula groups.
   - The v4n stack combines a FPNet fingerprint model, a 160-feature LightGBM ranker, "moderate" class-3 generation, a gated PubChem channel, and ICEBERG + GLACIER re-scoring within same-formula groups.
   - The second engine is the prvsiyan/megayak two-ranker with ChEBI + LIPID MAPS.
3. **The public-LB noise floor is about ±0.006–0.02** (FACT, several independent measurements; see §2.1).
   - The public LB is about 130 molecules (FACT: two independent statements; INFERENCE on the exact size).
   - Most single-lever gains reported between 0.33 and 0.40 are within 1–3× that noise.
4. **Levers that clearly beat the noise on the LB** (FACT):
   - per-spectrum + merged FPNet pair, +0.019;
   - the FPNet channel itself, +0.033 and then +0.036;
   - analog propagation, +0.082;
   - ICEBERG → GLACIER → forward weight 1.0 → fusion, 0.358 → 0.399 across the whole chain;
   - popularity prior + deeper PubChem screen, 0.373 → 0.386.
5. **Things that clearly hurt** (FACT):
   - dumping PubChem candidates into the pool (0.335 → 0.205);
   - a candidate cap of 80 (0.339 → 0.297; 0.335 → 0.282);
   - forward-model weight 2.0 (0.386 → 0.352);
   - premature tautomer dedup (0.339 → 0.327, CLAIM);
   - deeper or "honest-CV" rankers (0.335 → 0.321 / 0.330).
6. **Most remaining errors are same-formula isomers** (FACT, several authors):
   - alex chilton: 95.1% of the gap;
   - Udam Liyanage: the wrong winner has the same formula "about 98%" of the time;
   - DancingLumberjack: "Ranking beats recall about 17:1".
7. **Top teams (≥ 0.42) have disclosed nothing** (FACT: none posted). The best-ranked author who shared an approach is Udam Liyanage (rank 6, 0.414); see §4.

---

## 1. Notebook table

The columns are condensed. The main components and dependencies are summarised per notebook family further down (§1.2), so the table stays readable.

Notes on the table:
- "Votes" is the vote count on 2026-10-01.
- "Date" is the listing last-run date.
- "HW" is `machine_shape`: T4 means a GPU notebook, CPU means no GPU.
- All licences are "not shown via API, Apache 2.0 default (INFERENCE)" unless noted.

### 1.1 Table

| # | Notebook (link) | Author | Votes | Date | Stated public LB (exact) | Family / lineage | HW |
|---|---|---|---|---|---|---|---|
| 1 | [inversion/casmi-denovo-tutorial-notebook](https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook) | inversion (Kaggle staff) | 229 | 09-14 | not stated | official de novo tutorial (spectrum→SMILES transformer) | T4 |
| 2 | [haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline](https://www.kaggle.com/code/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline) (V17) | haideptry | 223 | 09-17 | "**0.300 - 0.336+**" (V17 row); others credit it "0.339" | 4-channel (prvsiyan-derived, no credit) | T4 |
| 3 | [prvsiyan/analog-propagation-casmi-2026-baseline](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) | prvsiyan | 116 | 09-21 | "0.335"; re-run "scored 0.328"; "family mean is 0.329 ± 0.004" | **root of the 4-channel family** | CPU |
| 4 | [evgendvorkin/enveda-casmi-2026](https://www.kaggle.com/code/evgendvorkin/enveda-casmi-2026) | Дворкин Е. В. | 106 | 09-30 | attributes "Public LB: 0.399" to the seyit pipeline it reproduces | reproduction of #8 | T4 |
| 5 | [beraterolelk/0-336-sota-enveda-casmi26-analog-ranker](https://www.kaggle.com/code/beraterolelk/0-336-sota-enveda-casmi26-analog-ranker) | Berat Erol ÇELİK | 79 | 09-29 | title "[0.336+ SOTA]", header "0.338+" — CLAIM | uncredited copy of early prvsiyan (INFERENCE) | CPU |
| 6 | [dmitriigluzdov/casmi-26-from-spectra-to-structures](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-from-spectra-to-structures) | Dmitrii Gluzdov | 61 | 09-30 | Version 19 "**0.386**"; V20 "0.374"; V18 0.373; V17 0.366; V14 0.326; V13 0.341. Current V21 not stated | Ahmed v4g/h + popularity prior + fusion | T4 |
| 7 | [malikhammadfarooq/casmi26-quad-channel-evidence-ranker](https://www.kaggle.com/code/malikhammadfarooq/casmi26-quad-channel-evidence-ranker) | Hammad Farooq | 55 | 09-17 | not stated (no markdown) | code copy of haideptry V17 (INFERENCE) | CPU |
| 8 | [seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399](https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399) | Seyit kaan Gunes | 46 | 09-30 | "**public LB 0.399**" | Ahmed v4n + two-ranker engine + RRF + ICE/GL on union | T4 |
| 9 | [denpugovkin/casmi-2026-protected-bio-db-tail](https://www.kaggle.com/code/denpugovkin/casmi-2026-protected-bio-db-tail) | Den Pugovkin | 42 | 09-30 | "public MRR@25 **0.347**" (for the byte-identical parent public-28); "This repackaging has not itself been scored" | W088 two-ranker + curated PubChemLite | T4 |
| 10 | [flexonafft/casmi26-confidence-gated-analog-generation](https://www.kaggle.com/code/flexonafft/casmi26-confidence-gated-analog-generation) | Igor Zharov | 38 | 09-30 | "Neither this notebook nor its Kaggle score has been verified yet." | dmitrii copy of v4g/v4i + popularity + fpnet_full1 (title ≠ content) | T4 |
| 11 | [ahmedberatozer/casmi26-v2-baseline](https://www.kaggle.com/code/ahmedberatozer/casmi26-v2-baseline) | Ahmed Berat Özer | 37 | 09-21 | not stated (INFERENCE: the "v1 pipeline … LB 0.354" engine) | **root of the Ahmed family** | T4 |
| 12 | [llccqq624/casmi26-next-direct-w088](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w088) | Jiachen Li | 36 | 09-20 | not stated; others quote W088 as "0.341" / "0.341+" | megayak two-ranker, w_pv 0.88 | T4 |
| 13 | [haideptry/0-350-casmi-2026-v32-ensemble](https://www.kaggle.com/code/haideptry/0-350-casmi-2026-v32-ensemble) | haideptry | 33 | 09-25 | title "0.350+", header "**Public LB Target:** 0.365+" — CLAIM/target | two-ranker + regio-isomer generator; isomer bank built from public test (INFERENCE: won't transfer) | T4 |
| 14 | [octaviograu/enveda-molecule-evidence-ranking](https://www.kaggle.com/code/octaviograu/enveda-molecule-evidence-ranking) | Octavi Grau | 32 | 09-15 | not stated | independent; entropy/cosine + prototype-FP + HGB | CPU |
| 15 | [josefreitasalvesneto/casmi-2026-a-complete-statistical-tour](https://www.kaggle.com/code/josefreitasalvesneto/casmi-2026-a-complete-statistical-tour) | José Freitas | 31 | 09-15 | EDA; proxy "MRR@25 ~ 0.91" (Class-1 proxy, CV) | EDA | CPU |
| 16 | [megayak/casmi26-two-rankers-one-engine](https://www.kaggle.com/code/megayak/casmi26-two-rankers-one-engine) | megayak | 30 | 09-17 | not stated here (predicted LB only) | **root of the two-ranker family** | CPU (~47 min) |
| 17 | [ahmedberatozer/casmi26-v3-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v3-inference) | Ahmed Berat Özer | 29 | 09-23 | "v1 pipeline … public LB 0.354"; PubChem-only "probe … 0.037"; v3.6 = "LB 0.358" (per v4f header) | Ahmed v3.6 | T4 |
| 18 | [ahmedberatozer/casmi26-v4l-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4l-inference) | Ahmed Berat Özer | 26 | 09-29 | not stated here; v4n header: "(v4l, LB 0.380)" | Ahmed v4l | T4 |
| 19 | [kozykappa/enveda-0-37-lb-score](https://www.kaggle.com/code/kozykappa/enveda-0-37-lb-score) | Kydyrbek Kozykorpesh | 25 | 09-28 | title "0.37 lb score" | code-identical copy of Ahmed v4g (INFERENCE) | T4 |
| 20 | [ahmedberatozer/casmi26-v4m-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4m-inference) | Ahmed Berat Özer | 23 | 09-29 | not stated here; v4n header: "(v4m, LB 0.378)" | Ahmed v4m | T4 |
| 21 | [avikdas567/physics-informed-spectral-transformer-retrieval](https://www.kaggle.com/code/avikdas567/physics-informed-spectral-transformer-retrieval) | Avik Das | 23 | 09-16 | not stated; "≈ 0.165 MRR" is an estimate | independent: cosine library + de novo transformer | T4 |
| 22 | [ektarr/baseline-enveda-casmi-2026](https://www.kaggle.com/code/ektarr/baseline-enveda-casmi-2026) | Maxim | 22 | 09-15 | not stated | Class-1 library search only | CPU |
| 23 | [ahmedberatozer/casmi26-v4n-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4n-inference) | Ahmed Berat Özer | 19 | 09-30 | not stated for v4n; seyit gives "v4n alone … 0.384" | Ahmed v4n = v4l + v4m | T4 |
| 24 | [ahmedberatozer/casmi26-v4h-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4h-inference) | Ahmed Berat Özer | 19 | 09-29 | not stated | Ahmed v4h (+GLACIER) | T4 |
| 25 | [ahmedberatozer/casmi26-v4g-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4g-inference) | Ahmed Berat Özer | 19 | 09-28 | not stated; dmitrii dataset card: "displays a 0.366 public Code score as of 2026-09-28" | Ahmed v4g | T4 |
| 26 | [seyitkaangunes/casmi26-e5c-union-gl-a06](https://www.kaggle.com/code/seyitkaangunes/casmi26-e5c-union-gl-a06) | Seyit kaan Gunes | 18 | 09-29 | not stated; #8 says the v4g-based chain reached "0.386 (union)" | v4g + two-ranker engine + RRF; FIORA embedded but `if False:` | T4 |
| 27 | [ahmedberatozer/casmi26-v4f-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4f-inference) | Ahmed Berat Özer | 18 | 09-26 | not stated | Ahmed v4f (+ICEBERG) | T4 |
| 28 | [ahmedberatozer/casmi26-v4q-inference](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4q-inference) | Ahmed Berat Özer | 1 | 09-30 | not stated | v4n + e5c fusion (Ahmed's own version of #8) | T4 |
| 29 | [navazshfathi/casmi-26](https://www.kaggle.com/code/navazshfathi/casmi-26) | navazsh fathi | 0 | 09-30 | the same version table as #6 (best "0.386") | INFERENCE: re-upload of #6, adds promotion rule | T4 |
| 30 | [bobthebot369/enveda-casmi-2026-v10-apex-fusion-sota](https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v10-apex-fusion-sota) | BobtheBot369 | 11 | 09-30 | not stated (team LB is 0.401 on the leaderboard; link to this notebook unproven) | v4n + popularity + two-ranker RRF + ICE/GL | T4 |
| 31 | [matterhorn3838/evenda-casmi-inferencesupernext](https://www.kaggle.com/code/matterhorn3838/evenda-casmi-inferencesupernext) | matterhorn3838 | 1 | 09-30 | not stated | v4f/g + e5c-style fusion | T4 |
| 32 | [thivnti/casmi26-apex-sovereign-fusion-v51](https://www.kaggle.com/code/thivnti/casmi26-apex-sovereign-fusion-v51) / [-grandmaster](https://www.kaggle.com/code/thivnti/casmi26-apex-sovereign-grandmaster) | Thái Văn Tài | 0 | 09-30 | not stated; "> 0.425" is a *target*; v51 header ≠ code (0.5/0.5 in code) | fusion / popularity forks | T4 |
| 33 | [wangpenghua/casmi26-v21-v4i-pop](https://www.kaggle.com/code/wangpenghua/casmi26-v21-v4i-pop) (+ [v22](https://www.kaggle.com/code/wangpenghua/casmi26-v22-v4m-pop), [v24 no-GL](https://www.kaggle.com/code/wangpenghua/casmi26-v24-v4i-pop-nogl), [v26 mu0.2](https://www.kaggle.com/code/wangpenghua/casmi26-v26-v21-mu02)) | wang penghua | 4 | 09-29/30 | not stated | **origin of the PubChem popularity prior** | T4 |
| 34 | [nursrijan/forensics-lab-dual-shield-two-ranker-sota](https://www.kaggle.com/code/nursrijan/forensics-lab-dual-shield-two-ranker-sota) | Nur Srijan | 5 | 09-29 | "(0.380+ Architecture)" — CLAIM | Ahmed v4m + tail fill | T4 |
| 35 | [raunakdey07/casmi-26-two-ranker-adduct-shifted-engine](https://www.kaggle.com/code/raunakdey07/casmi-26-two-ranker-adduct-shifted-engine) | Raunak Dey | 15 | 09-26 | "The score improvement from 0.341 to 0.342 …" | W088 + more seeds, depth 7 | T4 |
| 36 | [megayak/casmi26-one-engine-nine-scores](https://www.kaggle.com/code/megayak/casmi26-one-engine-nine-scores) | megayak | 0 | 09-19 | "The public scores span **0.312 – 0.337**" (9 submissions) | two-ranker ablation table | CPU |
| 37 | [franciscoangulo/enveda-casmi-v57-v54-metric-dedup](https://www.kaggle.com/code/franciscoangulo/enveda-casmi-v57-v54-metric-dedup) | Francisco Angulo | 0 | 09-23 | "v39 (0.339)"; "v53 (0.342)" | V17 family + BRICS tail (disabled in v57) | CPU |
| 38 | [alexchilton/casmi26-mist-blend-msbuddy-v65](https://www.kaggle.com/code/alexchilton/casmi26-mist-blend-msbuddy-v65) / [genll-tail-v58](https://www.kaggle.com/code/alexchilton/casmi26-genll-tail-v58) | alex chilton | 0–2 | 09-21/22 | quotes "v39 (0.339)", "v53 (0.342)"; own not stated | V17 family + MIST blend / BRICS tail | CPU |
| 39 | haideptry V18/V19/V22–V27 ([V22](https://www.kaggle.com/code/haideptry/casmi26-v22-pure-v17-deep-ranker), [V23](https://www.kaggle.com/code/haideptry/casmi26-v23-longitudinal-sota), [V27](https://www.kaggle.com/code/haideptry/casmi26-v27-tautomer-adduct-sota), [mastering…](https://www.kaggle.com/code/haideptry/casmi26-mastering-the-domain-gap-two-rankers)) | haideptry | 1–3 | 09-17…22 | V22 "0.333" (quoted in V23). Everything else is targets: V18 "0.341+", V19 "0.343 - 0.345+", V23 "0.345+", two-ranker "0.353+", V24–V26 "> 0.360", V27 "> 0.380" | V17 family | T4 |
| 40 | [llccqq624/casmi26-20260919-anchor-0339](https://www.kaggle.com/code/llccqq624/casmi26-20260919-anchor-0339) | Jiachen Li | 1 | 09-19 | inherited "Public LB: 0.339" (V17 verbatim, includes the cap-80) | V17 repro | T4 |
| 41 | Forks with no own LB stated: [catsfunny w088](https://www.kaggle.com/code/catsfunny/casmi26-next-direct-w088), [llccqq624 w080](https://www.kaggle.com/code/llccqq624/casmi26-20260919-two-ranker-w80)/[w084](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w084)/[w092](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w092)/[w100](https://www.kaggle.com/code/llccqq624/casmi26-next-direct-w100), [hosen42](https://www.kaggle.com/code/hosen42/casmi26-two-ranker-engine-variants-hosen42-v2), [cacinie](https://www.kaggle.com/code/cacinie/casmi26-engine-v2), [bobthebot V5](https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v5-sota-ranker)/[V8](https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v8-frontier-sota)/[V9](https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v9-frontier-sota), [xhhuang v35](https://www.kaggle.com/code/xhhuang/enveda-casmi-v35-sota-engine)/[v36](https://www.kaggle.com/code/xhhuang/enveda-casmi-v36-clean-champion-engine) ("0.380+" CLAIM), [uninhibitedscholar ×3](https://www.kaggle.com/code/uninhibitedscholar/casmi26-v32-two-ranker-postprocessor-cpu-v2), [xolotlmictlan r32 ×3](https://www.kaggle.com/code/xolotlmictlan/casmi26-r32-rrf-w1k1), [qiuqiuh](https://www.kaggle.com/code/qiuqiuh/casmi26-analog-prop-v6-xadduct), [heon29](https://www.kaggle.com/code/heon29/casmi26-measured-fork), [chuanyuchangbio](https://www.kaggle.com/code/chuanyuchangbio/casmi26-pipeline-v1), [shreyashautomation ×3](https://www.kaggle.com/code/shreyashautomation/casmi26-false-c1-rescue), [n10705013](https://www.kaggle.com/code/n10705013/casmi26-4ch-tautomer-dedup-chebi-fork), [tamerlanomralinov](https://www.kaggle.com/code/tamerlanomralinov/casmi-tamerlan-baseline), [soukeaizenz](https://www.kaggle.com/code/soukeaizenz/chimera-tworanker-blend-casmi26), [matweyisupov ×2](https://www.kaggle.com/code/matweyisupov/casmi26-v3-hybrid-static-fallback) | various | 0–15 | 09-17…30 | none stated (W80 described only as "scored") | forks of the families above | mixed |
| 42 | Analysis notebooks: [kitopl data pitfalls](https://www.kaggle.com/code/kitopl/casmi-2026-data-pitfalls-that-break-your-cv), [dariushafshar ppm window](https://www.kaggle.com/code/dariushafshar/casmi26-your-ppm-window-drops-121-805-spectra), [dariushafshar keys](https://www.kaggle.com/code/dariushafshar/casmi26-10-4-of-keys-change-0-48-cost-you), [wguesdon](https://www.kaggle.com/code/wguesdon/casmi-2026-library-plus-fingerprint-ranking) | various | 4–9 | 09-19…21 | kitopl library-only "scored 0.151"; wguesdon 0.146 / 0.227 / 0.248 | data facts (§3.4) | CPU |

**Notebooks with a stated, actually-scored LB ≥ 0.33** (FACT; titles and targets excluded):

| Score | Pipeline | Source |
|---|---|---|
| 0.399 | seyit v4n + fusion + union | #8 |
| 0.386 | dmitrii V19 | #6, #29 |
| 0.386 | seyit v4g-based union | #8 |
| 0.384 | Ahmed v4n alone | quoted in #8 |
| 0.382 | seyit v4g + fusion | #8 |
| 0.380 | Ahmed v4l | quoted in #23 |
| 0.378 | Ahmed v4m | quoted in #23 |
| 0.374 | dmitrii V20 | #6 |
| 0.373 | dmitrii V18 | #6 |
| 0.366 | Ahmed v4g / dmitrii V17 | #25 dataset card, #6 |
| 0.358 | Ahmed v3.6 | #17 |
| 0.354 | Ahmed v1/v4b | #17 |
| 0.350 | seyit two-ranker engine + BIO | #8 |
| 0.347 | denpugovkin public-28 | #9 |
| 0.342 | francisco v53 | #37 |
| 0.342 | raunakdey | #35 |
| 0.341 | W088 (quoted) | #9 |
| 0.341 | dmitrii V13 | #6 |
| 0.339 | haideptry V17 | #40, #37 |
| 0.337 | megayak two-ranker 0.65/0.35 | #36 |
| 0.337 | starkhushi 4-model FP ensemble | D/742055 |
| 0.335 | prvsiyan | #3 |
| 0.333 | megayak, prvsiyan ranker alone | #36 |
| 0.333 | haideptry V22 | #39 |
| 0.331 | megayak, blend 0.80 | #36 |
| 0.330 | prvsiyan v8 / v15 | #3 |

### 1.2 Pipeline families: components and dependencies

There are four code families. Almost every high-scoring notebook is a fork or combination of them.

**(A) prvsiyan "Analog Propagation" 4-channel** (#3). Forks include haideptry V17 (#2), #5, #7, #37, #38, #39, #40, qiuqiuh, heon29 and chuanyuchangbio.
- *Candidates.* Train structures ∪ COCONUT 2.0 give 711,705 structures, deduped on InChIKey14.
  - Window ±10 ppm with a 30 ppm fallback. V17 uses 8.5 ppm.
  - ChEBI/LIPID MAPS is off by default (`USE_BIO_DB = False`). PubChem is off (`PC_TOPK = 0`).
- *Channels.*
  1. Spectral-entropy library match.
  2. Mass-shifted analog propagation, `Σ sim^p · Tanimoto`, with N_ANALOG 80–200 and p = 3 (V17 uses 4). Analog representatives are matched by instrument.
  3. MetFrag-lite, 1–2 bond breaks with ±2 H.
  4. FPNet: a 6-layer transformer from spectrum to a 6,930-bit fingerprint (ECFP4 ‖ ECFP6 ‖ RDKitFP ‖ MACCS), scored as `f·z`. A per-spectrum model (s2) and a merged-spectrum model (m1) are averaged.
- *Ranker.* 31 features into sklearn `HistGradientBoosting` (depth 6, 500 iterations), bagged over 2 class priors (0.30, 0.60) × 4 seeds. It is fitted at runtime on the simulated rows in `rank_train.npz`.
- *Not used.* No forward model and no generation.
- *Runtime.* CPU. MetFrag-lite takes about 14 ms per candidate. FPNet training "needs a GPU and ~2 h" (done offline).
- *Dependencies.*

  | Dataset | Licence |
  |---|---|
  | prvsiyan/casmi26-fp-models-v2 | CC0 |
  | prvsiyan/casmi26-ranker-features | CC0 |
  | prvsiyan/coconut-casmi26-candidates | CC BY 4.0 |
  | prvsiyan/chebi-lipidmaps-casmi26 | **CC BY-NC-SA 4.0** |
  | prvsiyan/pubchem-npformula-massindex | CC0 |
  | prvsiyan/rdkit-wheel-offline | Other |
  | aidensong123/casmi26-offline-rdkit-2026033 | "Other", empty description |
  | thedevastator/open-source-natural-product-annotations | CC0 |

**(B) megayak "Two Rankers, One Engine"** (#16). Forks include W080–W100 (#12, #41), raunakdey (#35), denpugovkin (#9), haideptry two-ranker/V32/V33 (#13, #39), cacinie, hosen42, bobthebot V8/V9 and tamerlanomralinov.
- The engine is family A plus:
  - an adduct-shifted library match;
  - same-polarity analogs;
  - adduct-aware MetFrag-lite;
  - metric-exact (tautomer-canonical) dedup of the final 25.
- *Ranker A.* prvsiyan's 31 features with the FPNet channel, 8 HGB models.
- *Ranker B.* 51 features without the FP channel, trained on 2,250 simulated molecules (`megayak/casmi26-simulated-ranker-rows`, CC0).
- *Blend.* A within-molecule rank blend with `w_pv` = 0.65 in the original, 0.88 in W088 and later.
- *Runtime.* CPU, "~47 min".
- *Additions in forks:*
  - PubChemLite in the pool: denpugovkin, curated `annotation_count >= 2`, 90,816 structures; PubChemLite v2.9.0 is CC BY 4.0 per denpugovkin, and franciscoangulo's derived set is CC0.
  - Regio-isomer generation: haideptry V32/V33, cacinie, uninhibitedscholar.
  - A gated PubChemLite channel with three-ranker RRF: bobthebot V8/V9.

**(C) Ahmed Berat Özer v1–v4 stack** (#11, #17, #18, #20, #23–25, #27, #28). Forks include kozykappa (#19), dmitriigluzdov (#6, #29), flexonafft (#10), wangpenghua (#33), nursrijan (#34) and bobthebot v10 (#30).
- *Engine* (the `casmi` package, shipped inside the datasets):
  - pool = train ∪ COCONUT (710,701 rows);
  - FPNet A as the engine bank (`fpnet_full1` from v4m on);
  - class-3 generation `EngineCfg(generate=True)`, which "can also derive a new mass-compatible structure from a close spectral analogue". The internals are not documented in any notebook (INFERENCE: the code is in `casmi26-v4b-models`);
  - fe_v4 feature families: "derivation evidence / class-3 priors, fragmentation 2.0, analog-structure relations, DreamsFP views";
  - 160 features into 4 averaged LightGBM boosters, trained on the simulation "sim1x";
  - keeps the top 60.
- *PubChem-only channel* (a subprocess with a 4 h timeout):
  - `ahmedberatozer/casmi26-pubchem-tier`: 105.9M structures, stereo-stripped, CHNOPS + halogens, 150–1250 Da, 7.2 GB;
  - ±10 ppm window;
  - pass 1 is a partial ECFP4-block `f·z` to the top `PC_N1` = 1000 (5000 from v4i); pass 2 is the full `f·z`;
  - keys already in the pool are dropped, then the top 25 are kept.
- *Gate.*
  - If `lib_max ≥ 0.9`, the list is left untouched.
  - Otherwise, if (best PubChem f·z − best pool f·z) > 600, PubChem gets slots [2,4,6,8,10]; else slots [4,8,12,16,20].
  - The docstring says: "Gate chosen on out-of-fold simulation (positive under all mixtures)."
- *Forward models* (in subprocesses on RDKit 2025.03.6):
  - ICEBERG 2.1 from ms-pred (MIT), MassSpecGym `msg_all` checkpoint (`ahmedberatozer/casmi26-iceberg`);
  - GLACIER, MassSpecGym checkpoint (`ahmedberatozer/casmi26-glacier`);
  - within same-formula groups: `z(ranker) + λ·z(ICE) + λ·z(GL)`, with λ = 0.5 until v4l and 1.0 from v4l;
  - budgets 2700/2400 s in v4, 5400/3600–4000 s in the fusion notebooks.
- *Fusion* (e5c/#8/v4q): weighted RRF `1/(3+r_v4) + 0.6/(3+r_engine)` over the top 40, then ICE/GL again on the fused union.
- *Popularity prior* (wangpenghua v21 → dmitrii V19):
  - `pop = log1p(PubChem SID) + log1p(PMID)`;
  - PubChem list: `z(f·z) + 0.25·pop`, plus a union of the 200 most-popular structures in the window;
  - main list: `z(ranker) + 0.15·pop` for pool structures only; generated candidates keep their slots.
- *Promotion rule* (dmitrii / navazshfathi, #29): promote the best PubChem proposal to rank 1 if `S = z(FPNet) + 0.25·pop > 6` and `pop ≥ 5`; use early slots if S > 5.
- *Hardware.* T4 GPU, internet off. Runtime "~3.5–4.5 h on T4 for the hidden test" (#8).
- *Dependencies.*

  | Dataset | Licence |
  |---|---|
  | `ahmedberatozer/casmi26-{v2-pool, v2-models, v3-models, v4b-models, fpnet-full1}` | "Other": "derived from … competition training data (which includes CC BY-NC licensed components) and COCONUT (CC BY 4.0) … non-commercial, with attribution" |
  | `casmi26-pubchem-tier` | Other (NCBI public data) |
  | `casmi26-iceberg`, `casmi26-glacier` | Other; the ms-pred code is MIT and the checkpoints are MassSpecGym |
  | `seyitkaangunes/casmi26-glacier` | Other; MIT, "MassSpecGym contrastive checkpoint" |
  | `metric/rdkit-2026-3-3-wheel` | **Unknown** |
  | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | Other: PubChem free reuse; LOTUS CC0; NPAtlas CC BY 4.0 |
  | `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets` | Other (mixed); hash-pinned mirror, 11 GB |

**(D) Independent / other.**
- #1 official de novo tutorial:
  - encoder/decoder transformer, 6+6 layers, d = 768, BPE SMILES tokens;
  - 200k spectra, 1 epoch, about 1 h 40 min of training on a T4;
  - 25 samples per spectrum, deduped on InChIKey14;
  - no candidate database, and no LB stated.
- #14 octaviograu: entropy/cosine similarity, a prototype fingerprint from spectral neighbours, and an HGB ranker exported to JSON. Its attached datasets can no longer be resolved.
- #21 avikdas: cosine library search plus a de novo transformer.
- #22 ektarr: source-weighted library search only.
- wguesdon: own peak-set transformer, 0.248.
- soukeaizenz: MIST-CF / MIST-fp / ICEBERG from unresolvable datasets, so the licence is unknown.

---

## 2. Ablation ledger (every "X changed LB from A to B" statement found)

"LB" = public leaderboard, as stated by the author. Negative results are marked **NEG**. Dates are the notebook listing date or the post date (2026). CV rows are kept separate (§2.3).

### 2.1 Noise-floor measurements (read first)

| Statement (exact) | Numbers | Source | Date |
|---|---|---|---|
| "re-run of the 0.335 row, changing nothing \| it scored 0.328"; "family mean is 0.329 ± 0.004" | 0.335 vs 0.328 | [prvsiyan #3](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) | 09-21 |
| "I submitted the same notebook version twice and got 0.292 and 0.298" | 0.292 / 0.298 | prvsiyan #3; also D/742088 (Victor alexandre) | 09-19/21 |
| "Rows 3 → 4 change nothing but seeds ... −0.017. That is the size of a seed draw here"; "treat any public score inside ±0.02 of another as a tie" | 0.337 vs 0.320 | [megayak nine-scores](https://www.kaggle.com/code/megayak/casmi26-one-engine-nine-scores) | 09-19 |
| "our 7-set ranker only … (submitted 3×, identical file) \| 0.272 / 0.272 / 0.272"; "Identical files score identically" | 0.272 ×3 | megayak nine-scores | 09-19 |
| Resubmitting the same notebook moves the score "~0.006" | ±0.006 | [D/742055](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742055) starkhushi | 09-19 |
| Offline gains "below about 0.03 were invisible on the public leaderboard"; SE of a two-submission difference "around 0.016"; "~130 public molecules"; "one test molecule is worth about 0.0025 MRR" | — | [D/743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254) Udam Liyanage (rank 6) | 09-25/29 |
| "A single molecule moves it by up to 0.0076, so differences around 0.005 are noise"; public LB ≈ 33% of the hidden set (~130 molecules) | — | [kitopl](https://www.kaggle.com/code/kitopl/casmi-2026-data-pitfalls-that-break-your-cv) | 09-21 |
| "Reading: sum of reciprocal ranks of PubChem-only truths / ~130 public molecules" | — | [Ahmed v4g pc_runner](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4g-inference) | ≤09-28 |

INFERENCE: with about 130 public molecules, one molecule moving from rank 2 to rank 1 is worth about 0.004. The ±0.006–0.017 seed spread is 2–4 molecules. The 0.0025/molecule figure in D/743254 corresponds to 400 molecules and conflicts with "~130" (see §5).

### 2.2 LB ablations

**Family A: prvsiyan 4-channel, and haideptry V-series**

| Change | A → B | Source | Date |
|---|---|---|---|
| Spectral library search only | 0.151 | [prvsiyan #3](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) | 09-21 |
| + analog propagation & calibrated ranker | 0.151 → 0.233 | prvsiyan #3 | 09-21 |
| + leaderboard-calibrated class weighting | 0.233 → 0.245 | prvsiyan #3 | 09-21 |
| + in-silico fragmentation (MetFrag-lite) | 0.245 → 0.266 | prvsiyan #3 | 09-21 |
| + spectrum→fingerprint model (f·z) | 0.266 → 0.299 | prvsiyan #3 | 09-21 |
| + merged-spectrum model, seed-averaged ranker | 0.299 → 0.335 | prvsiyan #3 | 09-21 |
| **NEG** + constants copied from a popular fork ("one of them was a recall bug": cap 80 by mass) | 0.335 → 0.282 | prvsiyan #3 | 09-21 |
| removing the candidate cap, 10 ppm window | 0.282 → 0.311 | prvsiyan #3 | 09-21 |
| "m1@36k merged alone" vs "m1@24k + s2" model pair: "+0.019 for the two-model channel, isolated" | 0.311 → 0.330 | prvsiyan #3 | 09-21 |
| **NEG** + a third model in the ensemble | 0.335 → 0.329 | prvsiyan #3 | 09-21 |
| **NEG** narrow priors (.40,.45,.50) + depth 10 | 0.335 → 0.330 | prvsiyan #3 | 09-21 |
| **NEG** + PubChem pool expansion, top-50 isomers by f·z | **0.335 → 0.205** | prvsiyan #3 | 09-21 |
| **NEG** depth 10, original priors (query-held-out CV choice) | 0.335 → 0.321 | prvsiyan #3 | 09-21 |
| formula-mass library index + instrument-matched analogs | 0.328 → 0.327 | prvsiyan #3 | 09-21 |
| N_ANALOG re-swept to 200 | 0.327 → 0.330 | prvsiyan #3 | 09-21 |
| **NEG / noise** + ChEBI + LIPID MAPS ("inside the noise band") | 0.299 → 0.295 | prvsiyan #3 | 09-21 |
| **NEG** targeted derivative enumeration (±O, ±CH₂, ±hexose …) | 0.337 → 0.335 | prvsiyan #3; D/742055 starkhushi | 09-19 |
| prvsiyan → haideptry: "PPM_WIN 10 → 8.5, N_ANALOG 80 → 100, SIM_POWER 3 → 4 — plus a 31-feature ranker" | 0.335 → 0.339 | [evgendvorkin #4](https://www.kaggle.com/code/evgendvorkin/enveda-casmi-2026) | 09-30 |
| haideptry V10–11 → V14 (confidence gate C1_HI 0.60) → V15 → V16 → V17 | 0.163 → 0.193 → 0.205 → 0.266 → "0.300 - 0.336+" | [haideptry #2](https://www.kaggle.com/code/haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline) | 09-17 |
| **NEG (CLAIM)** "Trap 1": CAND_CAP 80 "eliminated 24.8% of true Class-2 reachable structures" | 0.339 → 0.297 | [haideptry mastering](https://www.kaggle.com/code/haideptry/casmi26-mastering-the-domain-gap-two-rankers), copied in W080/W088 | 09-18 |
| **NEG (CLAIM)** "Trap 2": premature tautomer dedup "dropped 1,309 valid V17 candidate positions" | 0.339 → 0.327 | same | 09-18 |
| **NEG** V22: depth 10 + no cap | 0.339 → 0.333 | [haideptry V23](https://www.kaggle.com/code/haideptry/casmi26-v23-longitudinal-sota) | 09-19 |
| "Hedging Class-1/Class-2 trade-off improves LB by ~0.003" | +0.003 | haideptry #2 | 09-17 |
| **NEG** "making the model more aggressive about library matching" | 0.152 → 0.145 | [D/741597](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741597) haideptry | 09-16 |
| "Adding COCONUT retrieval and mass-shifted analog propagation" | 0.151 → 0.245 | D/741597 | 09-16 |
| francisco: v39 → v53 | 0.339 → 0.342 | [francisco v57](https://www.kaggle.com/code/franciscoangulo/enveda-casmi-v57-v54-metric-dedup) | 09-23 |
| francisco BRICS generation: "v54 (COCONUT) 0.050 against v52 (enveda-180) 0.046" (INFERENCE: generated-only or tail metric; context unclear) | 0.046 → 0.050 | francisco v57 | 09-23 |
| wguesdon: library only → + fingerprint model → + analog propagation; "a local gain of 0.041 became 0.021 on the leaderboard" | 0.146 → 0.227 → 0.248 | [wguesdon](https://www.kaggle.com/code/wguesdon/casmi-2026-library-plus-fingerprint-ranking) | 09-19 |

**Family B: megayak two-ranker**

| Change | A → B | Source | Date |
|---|---|---|---|
| prvsiyan's shipped ranker only, ±10 ppm | 0.333 | [megayak nine-scores](https://www.kaggle.com/code/megayak/casmi26-one-engine-nine-scores) | 09-19 |
| rank blend 0.65·prvsiyan + 0.35·ours, 2 seeds | 0.337 | megayak nine-scores | 09-19 |
| **NEG / noise** same, 4 seeds | 0.337 → 0.320 | megayak nine-scores | 09-19 |
| blend 0.80 | 0.320 → 0.331 | megayak nine-scores | 09-19 |
| **NEG** window recentred −1.45 ppm (np-examples bias): "cost -0.011 to -0.025 LB" | 0.312 / 0.329 / 0.322 at blends 0.65 / 0.80 / 1.00 | megayak nine-scores | 09-19 |
| leak-free FPNet for "ours" at 0.65 / 0.50 ("inside the noise band") | 0.320 → 0.327 / 0.326 | megayak nine-scores | 09-19 |
| **NEG** "ours" ranker alone, no FP channel | 0.272 | megayak nine-scores | 09-19 |
| W088 two-ranker → + curated PubChemLite (annotation_count ≥ 2) in pool. The author: "We do not infer a 9-point leaderboard lift" | 0.341 → 0.347 | [denpugovkin #9](https://www.kaggle.com/code/denpugovkin/casmi-2026-protected-bio-db-tail) | 09-30 |
| **NEG** "removing unique ChEBI from our earlier stack" | 0.332 → 0.315 | denpugovkin, quoted in [n10705013](https://www.kaggle.com/code/n10705013/casmi26-4ch-tautomer-dedup-chebi-fork) | 09-20 |
| W088 → seeds 4→8 (aux 2→6), depth 6→7 | 0.341 → 0.342 | [raunakdey #35](https://www.kaggle.com/code/raunakdey07/casmi-26-two-ranker-adduct-shifted-engine) | 09-26 |
| two-ranker engine + BIO (ChEBI + LIPID MAPS) "alone" | 0.341 → 0.350 | [seyit #8](https://www.kaggle.com/code/seyitkaangunes/casmi26-v4n-engine-fusion-union-lb-0-399) | 09-30 |

**Family C: Ahmed v-series, and dmitrii / seyit on top of it**

| Change | A → B | Source | Date |
|---|---|---|---|
| v1 pipeline (FPNet A + LightGBM + moderate class-3 gen) | 0.354 | [Ahmed v3 #17](https://www.kaggle.com/code/ahmedberatozer/casmi26-v3-inference) | 09-23 |
| PubChem-only list as a probe submission | 0.037 | Ahmed #17 | 09-23 |
| + gated PubChem-only channel (v3.6) | 0.354 → 0.358 | Ahmed #17, [v4f header](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4f-inference) | 09-26 |
| **NEG** v3.6 → v4b (fe_v4 families + sim1x LightGBM ranker) | 0.358 → 0.354 | v4f header | 09-26 |
| v4c (learned PubChem merge) and v4d (DreamsFP D2 engine) → "v4f: back to v4b config" (INFERENCE: NEG; no numbers) | — | dataset `ahmedberatozer/casmi26-v4b-models` version notes | 09-25/26 |
| + ICEBERG re-scoring (v4f/v4g) | 0.358 → 0.366 (v4g, per [dmitrii card](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets)); a kozykappa copy is titled "0.37" | dmitrii card, [#19](https://www.kaggle.com/code/kozykappa/enveda-0-37-lb-score) | 09-28 |
| + GLACIER (dmitrii V18) | 0.366 → 0.373 | [dmitrii #6](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-from-spectra-to-structures) | 09-30 |
| v4i (PubChem pass-1 1000 → 5000) | 0.373 (per #4's growth table) | [#4](https://www.kaggle.com/code/evgendvorkin/enveda-casmi-2026) | 09-30 |
| v4l: ICE_LAM = GL_LAM 0.5 → 1.0 ("+0.007 LB") | 0.373 → 0.380 | [v4n header](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4n-inference), #4 | 09-30 |
| **NEG** v4m: fpnet_full1 engine bank at λ = 0.5 | 0.378 (vs 0.380 for v4l) | v4n header | 09-30 |
| v4n = v4l + v4m. #4 states "fpnet_full1 instead of fpnet_0 — +0.013 LB (main lever)". INFERENCE: the +0.013 is not derivable from the quoted numbers (v4m 0.378 < v4l 0.380) | 0.384 | seyit #8, #4 | 09-30 |
| + deeper PubChem screen + popularity prior (dmitrii V19) | 0.373 → **0.386** | dmitrii #6 | 09-30 |
| **NEG** V19 → V20: promotion of documented PubChem proposals + unit forward weights (confounded) | 0.386 → 0.374 | dmitrii #6 | 09-30 |
| **NEG** V13 → V14: "Primary branch order plus the same proposal rule". The author: "That local advantage did not transfer" | 0.341 → 0.326 | dmitrii #6 | 09-30 |
| v4g base → + two-ranker engine RRF fusion → + forward models on the union | 0.366 → 0.382 → 0.386 | seyit #8 | 09-30 |
| v4n base → + fusion + union | **0.384 → 0.399** | seyit #8 (checked against local copy) | 09-30 |
| "Fusion of two engines … +0.006 LB" | +0.006 | #4 | 09-30 |
| **NEG** forward-model weight 2.0 everywhere: "too much forward weight destroys confident library matches" | 0.386 → 0.352 | seyit #8 | 09-30 |
| same-formula window TOPN 60 → 100 (no change) | 0.386 → 0.386 | seyit #8 | 09-30 |

**Discussion threads**

| Change | A → B | Source | Date |
|---|---|---|---|
| starkhushi pipeline: library → + COCONUT analog → + newer COCONUT → + gated library term → + in-silico fragmentation | 0.158 → 0.223 → 0.243 → 0.250 → 0.283 | [D/742055](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742055) | 09-19 |
| + popular fork settings (8.5 ppm, sim⁴, 100 analogs, W1 0.45, ChEBI/LIPID MAPS) on prvsiyan | 0.335 → 0.335 | D/742055 | 09-19 |
| + two self-trained FP models (4-model ensemble) | 0.335 → 0.337 | D/742055 | 09-19 |
| **NEG** own FP models *instead of* the originals | 0.335 → 0.330 | D/742055 | 09-19 |
| **NEG** two-ranker engine (0.65/0.35) as published by haideptry | 0.335 → 0.323 | D/742055 | 09-19 |
| **NEG** FPNet fine-tuned on timsTOF (offline top-1 improved) | 0.337 → 0.328 | D/742055 | 09-22 |
| Probe: only the library twin at rank 1 | 0.275 | [D/742088](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742088) Victor alexandre | 09-19 |
| Probe: only ranker's #2 at rank 1 | 0.055 | D/742088 | 09-19 |
| Probe: only ranks 3–25 | 0.059 | D/742088 | 09-19 |
| Full list; "perfect-rerank ceiling of that top-25 ≈ 0.40" | 0.324 | D/742088 | 09-19 |
| **NEG** filler/junk SMILES padding to 25: "scored the whole submission 0.000" (twice) | 0.000 | D/742088 | 09-19 |
| Reorder ranks 2–25 by Tanimoto-to-twin: "an exact tie" | 0.324 → 0.324 | D/742088 | 09-19 |
| PubChem as a full candidate source: "+0.002"; tail slots 16–25 only: "+0.001" | +0.002 / +0.001 | D/742088 DancingLumberjack | 09-19 |
| Aggregation feature: CV predicted "+0.006", LB moved "+0.008" | +0.008 | D/742055 DancingLumberjack | 09-19 |
| **NEG** fragmentation scorer tuned on the isomer panel: "ours moved 0.341 to 0.336, i.e. nothing" | 0.341 → 0.336 | [D/743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254) NeckBeard | 09-29 |
| **NEG** (second-hand) full-PubChem expansion | 0.335 → 0.205 | D/743254 (repeats prvsiyan) | 09-29 |
| **NEG** static candidate artifact keyed to visible-test masses: "scored identical to three decimal places" (silent no-op at grading) | = baseline | [D/744172](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744172) | 09-28 |
| **NEG** curated-DB membership flags "looked great on public holdouts and hurt on the leaderboard" (no numbers) | — | D/743254 Udam | 09-25 |
| **NEG** generated candidates "has not yet paid off on the leaderboard" | — | D/743254 Udam | 09-25 |

### 2.3 CV / proxy ablations that matter for design (not LB)

**Class-2 simulation** (CV, prvsiyan #3 and forks):

| Setting | Class-2 MRR |
|---|---|
| ±10 ppm | 0.521 |
| ±20 ppm | 0.509 |
| ±30 ppm | 0.500 |
| + all PubChem isomers | 0.350 |
| + top-10 PubChem isomers | 0.337 |
| Four-channel ranker: COCONUT + train | 0.732 |
| + top-25 PubChem | 0.379 |
| + top-100 PubChem | 0.328 |
| + top-500 PubChem | 0.338 |

Other prvsiyan CV results:
- Candidate cap, truth retention: none 0.992, cap 400 0.992, cap 200 0.952, cap 80 by mass 0.752.
- Candidate cap, predicted ΔLB −0.057 against −0.053 observed on the LB.
- Channels alone: frag 0.259, FPNet 0.468, analog 0.521, all four via the ranker 0.612.
- Within-isomer-group MRR: analog 0.5411, f·z 0.4904, frag 0.2779, lib 0.1175, random 0.1688.

**PubChem channel with the answer removed from the pool** (CV panel; dmitrii dataset cards, #6):

| Setting | MRR@25 / top-1 |
|---|---|
| FPNet only, shortlist 1,000 | 0.510 |
| shortlist 5,000 | 0.512 |
| whole window | 0.511 |
| + popularity prior 0.25 | **0.951** |
| FPNet only, top-1 | 40% |
| + popularity, top-1 | 93% |
| Merge rules: main list only | 0.361 |
| fixed slots | 0.519 |
| promotion S > 6 | 0.681 |
| S > 6 & pop ≥ 5 | 0.718 |

Caveat from the card itself: "Thresholds were chosen on these same molecules."

**FPNet training-exposure leak:**
- FPNet A: 0.934 MRR on training structures vs 0.581 on the unseen np-examples panel (42% top-1).
- fpnet_full1 (trained on the panel): 0.928 on the panel, 89% top-1.
- megayak: public FP weights give "0.49 (enveda-np-examples, genuinely held out) but 0.76–0.82" on the libraries they saw.
- A leak-free FPNet raised sim C2 from 0.655 to 0.689.

**Two-ranker features** (megayak #16; predicted LB = 0.162·c1 + 0.220·c2):

| Change | Result |
|---|---|
| ranker rows 819 → 2,250 molecules | C2 0.626 → 0.644 |
| + adduct-shifted library | C1 0.852 → 0.872, C2 0.644 → 0.655; predicted LB 0.280 → 0.285 |

**Negative (NEG) CV results:**
- DreaMS as an analog channel ties or loses: 0.58 vs 0.57–0.63 (megayak), 0.1555 vs 0.1129 (prvsiyan). Its truth-vs-blocker ordering is below chance on the pairs that matter (alex chilton, D/742055).
- Cross-encoder attempts all failed (prvsiyan).

**Same-formula oracle** (alex chilton, D/742055):
- The shipped system scores "0.7315". With same-formula blockers removed it scores "0.9869", which is "95.1% of the total gap".
- Seven attempts to fix this gained +0.0000 to +0.0037 each: MetFrag-lite, connectivity, formula gate, CE depth, neutral loss, ICEBERG (+0.0015), ring-cut (−0.00178).

**Fragmentation on the isomer panel** (NeckBeard, D/743254):
- "+0.065" when tuned, "+0.023 ± 0.007" on held-out panels ("tuning inflated the effect about 2.8x").
- The LB did not move (0.341 → 0.336).

**Other panels:**
- alex chilton MIST blend (w = 0.3) on 276 drug-like molecules: 0.2705 → 0.3139 (+0.0434, CI [+0.0137, +0.0743]). The natural-product rerun is "VOID: 209 of 216 are in MIST's training set" ([v65](https://www.kaggle.com/code/alexchilton/casmi26-mist-blend-msbuddy-v65)).
- NPAtlas adds "1,113 structures … and zero holdout truths". LOTUS is "97.5% already contained" (alex chilton, D/742055). Udam: ChEBI, LIPID MAPS and NPAtlas "added nothing beyond PubChem ∪ COCONUT".

---

## 3. Host rulings and clarifications

The hosts are identified by the API `author_type` field:
- **David Healey** (Enveda): HOST.
- **Marie Killian**: HOST.
- **inversion** and **Ashley Oldacre**: Kaggle ADMIN/staff.

All quotes are verbatim.

### 3.1 External data and candidate databases

- **General licensing** (Welcome post, David Healey, 09-14, [D/741359](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359)):
  > "Per the competition rules, the winning solution cannot include any licenses that would otherwise prohibit commercial use or redistribution. … Any external data, models, or tools you rely on must be compatible with that license." … "Solutions built on data or tools that cannot be commercially licensed will not be eligible to win, and we reserve the right to disqualify any submission that does not comply or can't be reproduced with allowed data and tools."
- **PubChem** (09-18, [D/741857](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741857)):
  > "PubChem structures for retrieval are acceptable for prize-eligible solutions."
- **COCONUT, including its non-commercial sub-collections such as FooDB** (09-25, [D/743234](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743234)):
  > "Yes, COCONUT's own license is sufficient."
- **CFM-ID spectra of COCONUT; LGPL tools** (09-19, [D/741912](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741912)):
  > "Spectra generated from COCONUT structures with CFM-ID is fine; compliant with the external data policy. 2.) LGPL-licensed runtime dependency is fine also."
- **Enveda-180 Zenodo release** (09-27, [D/743569](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743569)):
  > "Yes you can use the Zenodo release"
- **NIST** (09-29, [D/744470](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744470)):
  > "the NIST dataset is not permitted for solutions, any solution using or trained on NIST will be disqualified. It is not freely available and its license has restrictions on redistribution of itself or derivatives, including model weights."
- **Rules page**:
  - 2.6.b: "The use of external data and models is acceptable unless specifically prohibited by the Host." / "Purchasing a license to use a proprietary dataset that exceeds the cost of a prize … would not be considered reasonable."
  - 2.5.a.3: "In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license … for that data and/or model(s)."
  - Winner licence: "Open Source - MIT".

### 3.2 Pretrained models

- **Test for open weights** (09-18, [D/741844](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741844)):
  > "The main question with model weights is whether the license for the data in question allows redistribution (even noncommercial redistribution) of derivatives like model weights. If so, then you should be fine to use it."
- **Open weights and restricted data** (09-20 and 09-25, [D/742193](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742193)):
  > "We can't blanket approve all open weights models." … "Many datasets (MassSpecGym perhaps) are functionally open source … These are fine for solutions, as are any models trained on this competitions train.parquet … (enveda-np-examples we neglected to assign a specific license to but should also be considered open source)" … "There are also fully commercial datasets like METLIN and those distributed by the mass spec companies (Thermo, Bruker, SciEx). Also to avoid."
- **MassSpecGym-trained models such as ICEBERG 2.1 and MIST** (09-25, [D/742991](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742991)):
  > "we consider MassSpecGym and models trained on it and released with permissive licenses as fair game." … "(or proprietary commercial datasets like METLIN …) … But if they do exist they would not be eligible."
- **GLACIER** (09-29, [D/744338](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744338)):
  > "Yes, MassSpecGym is allowed, so is GLACIER"
- **CFM-ID 4, trained on METLIN** (09-27, [D/743774](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743774)):
  > "CFM ID 4 is fine. It's published by a major lab in a major journal, so we'll proceed with the assumption that the authors had permission to distribute the weights."
- **ChemBERTa** (09-19, [D/742011](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742011)):
  > "Yes, chemberta is open source"
- **Models trained on train.parquet** (09-25, [D/742265](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742265) and [D/743236](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743236)):
  > "any open source models trained on train.parquet are fair game."
  > "The Competition Data in train.parquet is a compilation of open-source databases, and so models trained on it can be open-sourced. Sorry for the licensing confusion on that; the **test** data when it is released will be restricted to noncommercial use."
  Private datasets plus rebuild scripts "seem acceptable".
- **Off-Kaggle training** (09-18/19, [D/741876](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741876)): training on the competition data in a private cloud or Hugging Face repo and bringing the weights back:
  > "Yes, you can do that."

### 3.3 Compute and runtime

- Code Requirements page:
  > "CPU Notebook <= 9 hours run-time" / "GPU Notebook <= 9 hours run-time" / "Internet access disabled" / "Freely & publicly available external data is allowed, including pre-trained models" / "Submission file must be named `submission.csv`"
- Marie Killian, 09-28 (D/742265):
  > "The P100s have been sunsetted."
- Kaggle staff give no individual support for failed reruns (D/742463).
- Rules: 5 submissions per day and 2 final selections. Team size is at most 5. Entry and merger deadline is 2026-12-07; final deadline is 2026-12-14.

### 3.4 Class definitions

- **Data page (FACT):**
  - "**1 — in public spectral libraries** | The structure has publicly available reference MS/MS spectra"
  - "**2 — known structure, no public spectra** | … the structure is in PubChem or the natural-products database COCONUT"
  - "**3 — novel structure** | The structure is not in PubChem at all and must be predicted de novo."
  - "the distribution over classes — and which molecule belongs to which class — is hidden for the duration of the competition."
- **Host, 09-23** ([D/742274](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274)):
  > "Class 3 is absent from both PubChem and COCONUT."
- **Participant estimates of class shares** (not host):

  | Source | Class 1 | Class 2 | Class 3 | Method |
  |---|---|---|---|---|
  | prvsiyan, early | ~16% | ~45% | ~39% | LB-probed |
  | haideptry, own arithmetic (D/741597) | — | ≈ 18% | — | LB-probed |
  | prvsiyan, later | ≈ 0.162 | 0.27–0.30 | ≈ 0.55 | "ceiling A + B ≈ 0.43" for retrieval-only |
  | kitopl | 0.151 / 0.8733 = **17.3%** | — | — | library-only LB score |
  | haideptry tiers | ~10–15% | ~45–55% | ~30–40% | — |

### 3.5 Public/private split

- The rules say only: "The 'Public Leaderboard' is a ranked display of Participants' Submission scores against a representative sample of the test data." **No host statement gives the percentage or the molecule count.**
- Host, 09-15 ([D/741404](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741404)):
  > "the public leaderboard score is calculated not on the contents of the test.parquet downloadable dummy file (which as you say just contains a sample of Enveda-180 spectra …) but rather on the public split of the Hidden test set provided to your submission notebook."
- Participant inferences:
  - "~130 public molecules" (Ahmed docstring; Udam; kitopl: "about 33%").
  - The hidden test has "~1,500 spectra of ~400 molecules, 1–16 spectra per molecule (median 3), all acquired on a Bruker timsTOF. Monoisotopic masses range from 157 to 1,159 Da (median 348)" (data page).

### 3.6 Metric

- **Evaluation page:**
  > "A prediction is correct when it describes the same **atom connectivity** as the answer. Both your SMILES and the answer are passed through RDKit's tautomer canonicalization (pinned at 2026.03.3) and reduced to the first block of their InChIKey (the InChIKey14)" … "you are not penalized for getting stereocenters or tautomer forms wrong."
  > "A submission is rejected if it is missing the `molecule_id` or `smiles` column, is empty, contains nulls in either column, repeats a `molecule_id`, or gives more than 25 semicolon-separated guesses"
  Metric notebook: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
- **Empty entries** (host, 09-23, D/742274):
  > "Empty entries are discarded before ranks are assigned. A;;B is read as two guesses, so B is at rank 2, not rank 3. A trailing ; is not a guess and does not affect your score"
- **Scorer version** (Marie Killian, 09-28, D/742265):
  > "The live scorer is identical to the published casmi-mean-reciprocal-rank v13. It has not changed. We are investigating the concerns raised in the thread you referenced." (That thread is D/742088: filler padding scored 0.000.)
- **Stereo in the answers** (host, 09-30, [D/744556](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744556)):
  > "Yes, solutions are stereo-stripped before canonicalization."
  - Participant evidence (Yohann Roux) in the same thread: written stereo can change the tautomer-canonical InChIKey14. INFERENCE: strip stereo from our predictions too.
- **Participant findings on keys:**
  - The shipped train `inchikey14` ≠ the scorer key for 1.58% of structures (kitopl) or ~4.4% of NP-library structures (D/742042).
  - 10.42% of COCONUT keys change under the scorer, but only 2.30% can move a score (dariushafshar).
  - Dedupe and join on the scorer key.

### 3.7 Data clarifications

- **Train update** (inversion, 09-15, [D/741471](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741471)):
  > "There has been a small update to the training data, adding water loss adducts to some of the samples."
- **CCS / ion mobility:**
  > "Its deliberately excluded; the test set has no CCS values." (David Healey, D/743368)
  > "It has the same columns as provided in test.parquet." (Marie Killian, D/743569)
- **Positive-mode m/z offset** (host, 09-29, [D/743395](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743395)):
  > "Tracking down which test spectra were processed with this version."
  The participant's claim in that thread: about +0.49 mDa in positive mode from using neutral Na in calibration.
- **Data page:**
  - "If your model consumes collision energy, use `collision_energy_ev`".
  - "Peaks above precursor + 2 Da were removed".
  - Ten test adducts are listed, but the column table says "one of the seven above" (see §5).

---

## 4. What top teams (LB ≥ 0.42) have disclosed

**FACT: nothing.**
- pikachu (0.464), Randy (0.435), Ozymandias31415 (0.425), Shehab Alshehabi (0.423) and chopper (0.421) do not appear as authors or commenters in any of the 58 topics read.
- A Kaggle `search_content` query returned nothing.
- No public notebook is attributable to them.

The closest proxy is **Udam Liyanage, rank 6, LB 0.414** ([D/743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254), 09-25, "Lessons from ~30 submissions"). His statements, quoted exactly:
- Candidate pool = PubChem ∪ COCONUT. PubChem is "Net positive for us, but only together with a ranking step that can handle same-formula decoys".
- When the truth is in the list but not first, the wrong winner has the same formula "about 98% of the time". The development metric is "a small same-formula isomer panel": "truth plus its best-scoring wrong same-formula candidates, capped at a few dozen".
- In-silico fragmentation is "where we found gains". The ranker is a GBDT (INFERENCE from the thread).
- ChEBI, LIPID MAPS and NPAtlas added nothing beyond PubChem ∪ COCONUT. Curated-DB membership flags "looked great on public holdouts and hurt on the leaderboard".
- Delete-truth novelty simulations "overstated reach about 3×". Generated candidates "has not yet paid off on the leaderboard", even when gated to predicted-novel molecules.
- timsTOF: "99% of precursor masses were within about 5 ppm"; "Going from 8.5 to 10 ppm left candidate counts essentially unchanged".
- Tautomers: "about 8% of molecules had a tautomer copy of the truth under a different PubChem key".
- For NP-chemistry changes, the timsTOF NP panel predicted the LB better than the class-2 public holdout.

Other signals (INFERENCE):
- hengck23 (Grandmaster) hints at training a SMILES→MS/MS forward model ([D/744343](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/744343)). A commenter says CASMI winners "generated 1TB or more training data".
- NeckBeard is in the 0.399 cluster; see his fragmentation ablation in D/743254.
- BobtheBot369 is 0.401 on the leaderboard. Their public v10 notebook is v4n + popularity + fusion, with no LB stated.

INFERENCE: the gap from 0.399 to 0.464 has not been explained in public. Udam's recipe (PubChem ∪ COCONUT + same-formula isomer ranking via fragmentation + GBDT) is the only disclosed approach above 0.40 that is not the public stack.

---

## 5. Open questions and contradictions between sources

1. **METLIN-trained weights.**
   - The host said on 09-25 that METLIN-trained models "would not be eligible" (D/742991) and are "to avoid" (D/742193).
   - On 09-27 he said "CFM ID 4 is fine" (D/743774), even though CFM-ID 4's stock models are trained on METLIN.
   - The stated test is whether the original authors had permission to distribute. NIST is treated far more strictly ("will be disqualified").
2. **Non-commercial terms.**
   - The Welcome post says non-commercial data or tools make a solution ineligible.
   - D/741844 accepts data whose licence allows "(even noncommercial redistribution)" of derivatives.
   - COCONUT is accepted despite its CC BY-NC sub-collections, and MassSpecGym despite its NC MassBank records.
   - The Rules page still lists the Competition Data as CC BY-NC 4.0, while the host says models trained on train.parquet can be open-sourced.
   - **Relevance:** `prvsiyan/chebi-lipidmaps-casmi26` is **CC BY-NC-SA 4.0**, and it is part of the 0.399 stack (the BIO component). The Ahmed pool, model and FPNet datasets say "non-commercial, with attribution". The RDKit metric wheel's licence is "Unknown". INFERENCE: forking the 0.399 stack as-is carries a prize-eligibility risk on the BIO dataset, not on the method.
3. **Stereo in predictions.** Answers are stereo-stripped (host). Whether predictions are stripped is not stated, and written stereo can change the key (D/744556).
4. **Filler, unparseable and duplicate guesses.**
   - The host says the scorer is v13, unchanged. A participant says v13 lets an unparseable guess consume a rank.
   - D/742088 reports that filler padding scored the whole submission 0.000. The host is "investigating".
   - Nobody has answered whether duplicate-key guesses consume ranks (D/741851).
   - Safe practice: no filler SMILES; dedupe on the scorer key.
5. **Public split size.**
   - "~130 molecules" (Ahmed docstring, kitopl "33%", Udam).
   - Udam's "one test molecule is worth about 0.0025 MRR" corresponds to 400 molecules, not 130 (1/130 ≈ 0.0077, which matches kitopl's "up to 0.0076").
   - No host number.
6. **Class shares.**
   - "16/45/39" (prvsiyan, early) vs f₂ "≈ 18%" (haideptry's own arithmetic) vs "0.27–0.30 / 0.55" (prvsiyan, later) vs Class 1 17.3% (kitopl).
   - All are LB-probed on about 130 public molecules. None is verified against the private split.
7. **PubChem helps or hurts.**
   - Hurts: prvsiyan 0.335 → 0.205 (pool expansion); CV Class-2 0.732 → 0.379.
   - Neutral: DancingLumberjack +0.002.
   - Helps as a gated channel: Ahmed 0.354 → 0.358; with the popularity prior, dmitrii 0.373 → 0.386. Udam: "net positive" only with an isomer ranker.
   - INFERENCE: the sign depends on how PubChem is merged (gate + fixed slots + popularity prior) and on isomer ranking. It does not depend on PubChem itself.
8. **The fpnet_full1 lever: gain or training exposure?**
   - #4 claims "+0.013 LB (main lever)", but Ahmed's own numbers give v4m (full1) 0.378 < v4l 0.380.
   - full1 was trained on enveda-np-examples, so that panel can no longer validate it (dmitrii: 89% vs 42% top-1).
   - dmitrii deliberately keeps FPNet A. Whether hidden-test chemistry overlaps np-examples is unknown.
9. **Fragmentation's value.**
   - Positive: Udam ("where we found gains"), starkhushi (0.250 → 0.283 on an early pipeline), prvsiyan (+0.021).
   - Near-zero: alex chilton ("free to delete"), NeckBeard (LB 0.341 → 0.336), starkhushi's 1-of-3 test (below chance).
   - INFERENCE: the value shrinks as the base pipeline gets stronger.
10. **Tautomer dedup.**
    - haideptry "Trap 2" says early dedup cost 0.339 → 0.327.
    - megayak, Udam, kitopl and xhhuang dedupe on the scorer key and report freed slots.
    - INFERENCE: the difference is *when* the dedup happens (before or after ranking, and with or without backfill). The unsupported number is the 0.327 (CLAIM).
11. **ppm window and calibration offset.**
    - prvsiyan: "+1.4 ppm systematic offset". Bryce Hedelius: +0.49 mDa in positive mode (host investigating).
    - megayak: recentring the window by −1.45 ppm *cost* 0.011–0.025 LB.
    - Window width: Udam says 8.5 vs 10 ppm barely matters. The CV prefers 10 ppm (predicted LB 0.361 vs 0.371). V17 used 8.5.
12. **Peak-noise statistic.** "82%" of test peaks below 1% (José Freitas, D/741404) vs "93%" (starkhushi, D/741641 and D/741423). The median of 18 peaks after a 1% floor agrees.
13. **v4g score.** "0.366" (dmitrii card) vs "0.37" (the kozykappa copy's title).
14. **Adduct count.** The data page lists ten test adducts; its column table says "one of the seven above". Seven appear in the public placeholder file.
15. **Visible-test leak threads** (D/741756, D/742278: 1,213/1,213 visible test spectra match enveda-180 train rows). The data page explains this: it is placeholder data taken from train. The host never replied in those threads, but separately confirmed that the LB uses the hidden test (D/741404).
16. **Test-leaking notebooks** (INFERENCE):
    - haideptry V33's isomer bank was "matched against 1,213 experimental test spectra".
    - matweyisupov embeds a cached submission for the public test hash.
    - Neither transfers to the hidden rerun, so their titles or scores say nothing about the method. Per D/744172, any artifact keyed to visible-test masses is a silent no-op at grading.
17. **Unanswered host questions:**
    - how the NIST ban is enforced for non-prize medalists (D/744470);
    - 22 identical spectrum pairs with conflicting labels (D/744446);
    - which test spectra carry the m/z offset (D/743395);
    - the exact public/private split.
18. **Untested public lines** (no LB anywhere):
    - GLACIER-predicted spectral library + RRF (xolotlmictlan r32);
    - FIORA (MSnLib v7, CC BY 4.0; embedded but disabled in e5c and matterhorn);
    - MIST blend (+0.043 local on drug-like molecules only);
    - dmitrii's rank-1 promotion rule on its own (confounded in V20: 0.386 → 0.374);
    - the fusion forks by bobthebot v10, matterhorn and thivnti.
