# Papers kept in the repo

Reference copies of papers a campaign implements from, kept here because they have no code
release and the implementation choices in `src/charts/` cite their equation numbers.

- `loink_2609.21275.pdf` -- Somerfield, Abood, Wang, Manchester, *LOInK: Learned Optimal Inverse
  Kinematics via Structured Neural Surrogate Models*, arXiv:2609.21275 (CC BY 4.0). The second
  chart family (`src/charts/loink.py`, branch `chart-loink`). No code online as of 2026-10-22.
- `bilipnet_2402.01344.pdf` -- Wang, Dvijotham, Manchester, *Monotone, Bi-Lipschitz, and
  Polyak-Lojasiewicz Networks*, ICML 2024, arXiv:2402.01344 (CC BY 4.0). The building block
  (BiLipNet, the Davis-Yin inversion). Official JAX/Flax code: https://github.com/acfr/PLNet (MIT),
  not vendored; our torch port is checked against it (`tests/data/loink_reference/`).
