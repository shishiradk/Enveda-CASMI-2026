# Research Notes: Modified Cosine Spectral Similarity (matchms 0.33.1)

## 1. matchms Version Inspected
*   **Version:** 0.33.1
*   **Source Location:** `research/_matchms_src_ref/` (copied from `C:\Users\LENOVO\anaconda3\Lib\site-packages\matchms\similarity\`)

## 2. THEORETICAL Modified Cosine
The Modified Cosine similarity is an extension of the Cosine similarity used in mass spectrometry to account for chemical modifications or different substructures that result in a mass shift.

*   **Direct Matches:** Like standard cosine, it matches peaks that have the same m/z (within tolerance).
*   **Shifted Matches:** It also matches peaks where the m/z difference between two peaks is approximately equal to the difference in precursor m/z of the two molecules.
    *   Mathematically: $|(m/z_1 - precursor\_m/z_1) - (m/z_2 - precursor\_m/z_2)| \leq tolerance$.
    *   This is equivalent to matching the "neutral loss" of the fragment.
*   **Peak Assignment:**
    *   **Greedy:** Pairs are sorted by their contribution to the score and assigned one-by-one.
    *   **Optimal (Hungarian):** Pairs are assigned to maximize the total score using a bipartite matching algorithm (linear sum assignment).
*   **Weighting:** Intensities and/or m/z values are often raised to a power (e.g., $I^{1.0}$, $mz^{0.0}$) to adjust the influence of abundance vs mass.

## 3. matchms IMPLEMENTATION SPECIFICS

### Algorithm & Parameters
*   **Tolerance:** Default is **0.1**. This is interpreted as a Da (absolute) tolerance, not ppm, in the default classes.
    *   *Source:* `ModifiedCosineGreedy.py:34`, `ModifiedCosineHungarian.py:34`
*   **mz_power:** Default is **0.0**. This means m/z values do not contribute to peak weights by default.
    *   *Source:* `ModifiedCosineGreedy.py:34`, `ModifiedCosineHungarian.py:34`
*   **intensity_power:** Default is **1.0**. Peak weights are based linearly on intensity.
    *   *Source:* `ModifiedCosineGreedy.py:34`, `ModifiedCosineHungarian.py:34`

### Matching Logic
*   **Shift Calculation:** `mass_shift = precursor_mz_ref - precursor_mz_query`.
    *   *Source:* `ModifiedCosineGreedy.py:56`
*   **Candidate Search:** It collects both "zero pairs" (shift=0) and "nonzero pairs" (shift=mass_shift).
    *   *Source:* `ModifiedCosineGreedy.py:67-74`, `ModifiedCosineHungarian.py:67-74`
*   **Assignment Strategy:**
    *   **ModifiedCosineGreedy:** Concatenates all pairs, sorts them by weight (intensity product) in descending order, and greedily picks the best available matches.
        *   *Ties:* Uses `np.argsort(..., kind="mergesort")` which is stable. Ties in intensity product preserve their relative order from the concatenation of zero and shifted pairs.
        *   *Source:* `ModifiedCosineGreedy.py:82`, `spectrum_similarity_functions.py:98-103`
    *   **ModifiedCosineHungarian:** Builds a weight matrix where $W_{ij} = \max(\text{weight}_{ij, unshifted}, \text{weight}_{ij, shifted})$. It then uses `scipy.optimize.linear_sum_assignment` on `max_weight - weights` to find the globally optimal assignment.
        *   *Source:* `ModifiedCosineHungarian.py:84-133`
*   **Complexity:**
    *   **Greedy:** $O(N \log N)$ where $N$ is the number of candidate peak pairs.
    *   **Hungarian:** $O(M^3)$ where $M$ is the number of unique peaks involved in candidate pairs.
*   **Score Formula:** $Score = \frac{\sum (w_i \cdot w_j)}{\sqrt{\sum w_i^2} \cdot \sqrt{\sum w_j^2}}$ where $w = mz^{mz\_power} \cdot intensity^{intensity\_power}$.
    *   *Source:* `spectrum_similarity_functions.py:109`, `ModifiedCosineHungarian.py:147`

### Quirks & Gotchas
*   **Redundancy Check:** If the precursor mass difference is less than the tolerance (`abs(mass_shift) <= self.tolerance`), the class ignores the shift logic entirely and falls back to standard `CosineGreedy`/`CosineHungarian`.
    *   *Source:* `ModifiedCosineGreedy.py:58-63`
*   **Duplicate Pairs:** In the Hungarian version, if a peak pair matches both unshifted and shifted (e.g., if the mass shift is 0 or very small), it explicitly takes the maximum weight for that edge to avoid overcounting.
    *   *Source:* `ModifiedCosineHungarian.py:93-98`

## 4. WORKED TOY EXAMPLE

**Spectra Data:**
*   **Spectrum A (Reference):** Precursor m/z = **200.0**
    *   P1: (100.0, 50)
    *   P2: (120.0, 100)
    *   P3: (150.0, 40)
*   **Spectrum B (Query):** Precursor m/z = **210.0**
    *   Q1: (100.0, 60)
    *   Q2: (130.0, 90)
    *   Q3: (160.0, 30)

**1. Calculate Mass Shift:**
`mass_shift` = 200.0 - 210.0 = **-10.0**

**2. Find Matches (Tolerance = 0.1):**
*   **Direct (shift=0):**
    *   P1 (100.0) vs Q1 (100.0) -> **MATCH** (Weight = $50 \times 60 = 3000$)
*   **Shifted (shift=-10.0):**
    *   Check: $mz_{P} \approx mz_{Q} + mass\_shift$
    *   P2 (120.0) vs Q2 (130.0 - 10.0 = 120.0) -> **MATCH** (Weight = $100 \times 90 = 9000$)
    *   P3 (150.0) vs Q3 (160.0 - 10.0 = 150.0) -> **MATCH** (Weight = $40 \times 30 = 1200$)

**3. Assign Pairs (Greedy/Hungarian results are identical here):**
*   (P2, Q2) Weight = 9000
*   (P1, Q1) Weight = 3000
*   (P3, Q3) Weight = 1200
*   **Sum of Products:** 9000 + 3000 + 1200 = **13200**

**4. Normalization:**
*   $Norm_A = \sqrt{50^2 + 100^2 + 40^2} = \sqrt{2500 + 10000 + 1600} = \sqrt{14100} \approx 118.74$
*   $Norm_B = \sqrt{60^2 + 90^2 + 30^2} = \sqrt{3600 + 8100 + 900} = \sqrt{12600} \approx 112.25$
*   **Final Score:** $13200 / (118.74 \times 112.25) \approx 13200 / 13328.56 \approx \mathbf{0.9903}$
*   **Matches:** 3

## 5. Sources & Citations
*   `research/_matchms_src_ref/ModifiedCosineGreedy.py`
*   `research/_matchms_src_ref/ModifiedCosineHungarian.py`
*   `research/_matchms_src_ref/spectrum_similarity_functions.py`
*   Watrous et al. [PNAS, 2012, https://www.pnas.org/content/109/26/E1743] (Cited in matchms docstrings)
