# LOInK replication (arXiv 2609.21275, Sec. V and VII) with the torch port

2026-10-22, branch `chart-loink`. Port `src/charts/loink.py` (unchanged by this work). Scripts:
`scripts/loink_replication/{train_replication,planar3dof,hsa}.py`. Outputs (gitignored):
`results/loink_replication/{planar3dof,hsa}/` (figures, `summary.json` with every number below,
and a `runs` table: settings, final metrics, wall, `uptime` before/after each training run).
Every run: one process, CPU, 6 torch threads, under the 20G memory scope; 1-min load 0.5-10.5
before/after each training run.

## Parameters: given by the paper vs chosen here

| | Paper | Here (why) |
| --- | --- | --- |
| Planar data | 1e6 q ~ U([0, pi]^3), l = 1 m, y = p_x | same |
| Planar kinematics | "elbow up", q in [0, pi] | theta_1 = q_1, theta_i = theta_{i-1} - q_i (relative joints bend clockwise). Read off Fig. 3: its configurations are q ~ (2.03, 1.39, 1.50), (1.47, 1.05, 0.82), (0.46, 0.38, 0.31), which only this convention gives. The all-counter-clockwise formula is its mirror (q_1 -> pi - q_1, y -> -y) and puts the paper's targets in the sparse, high-cost end of the data: probe runs in `planar3dof/mirror_convention_probe/` |
| J_q, eq. 18 | -sum log(max(1 - q_hat^2, eps)), eps "small" | eps = 1e-3 |
| J_com, eq. 19 | (l/3) sum_{i=0}^{3} c^i_x^2 | c^0 = base (term identically 0), c^1..c^3 = link midpoints |
| lambda_com | not given | 1.0. With it the true constrained optima are q* = (1.96, 1.40, 1.57), (1.43, 1.08, 0.86), (0.39, 0.34, 0.27), which match Fig. 3's configurations; lambda_com = 0 gives (pi/2, pi/2, pi/2) at y = 1 (links not over the base). The 0.1 rerun was not needed |
| Net (planar) | 2 monotone layers, depth 4, width 256; offset MLP 128 | units 256x4, depth 2, head_hidden 128 |
| Net (HSA) | 2 layers x 2 hidden of 256; offset 64 | units 256x2, depth 2, head_hidden 64 |
| mu | 0.1 fixed, nu free | prod mu = 0.1 (sqrt(0.1) per layer), lognu learned, init nu = 2 per layer |
| Distortion penalty | "small" | 1e-3 log tau (1e-4 changed nothing on HSA) |
| lambda (eq. 16) | not given | **planar 0.01, HSA 0.1** (sweep below); lambda = 1 underfits the task |
| Optimiser | Adam 1e-3, batch 500, plateau patience 500 steps; 3 epochs (planar), HSA epochs not given | same; HSA 3 epochs. Plateau: factor 0.5, monitored on an EMA(0.99) of the training loss every step |
| Gradient clipping | not mentioned | off for planar; **global norm 1.0 for HSA** (without it a loss spike at step ~5800 left the EMA above its old minimum and the scheduler halved lr to 2e-7) |
| Input scaling | not given | network sees q_hat = (q - pi/2)/(pi/2) (planar), (u - c)/h normalised to [-1, 1] (HSA) |
| HSA task scaling | not given | y standardised per dimension (p_x sd 0.228 m, p_y 0.083 m, theta 0.875 rad); raw metres next to rad^2 costs make eq. 16's two terms incommensurate |
| HSA cost | J = abs(u - u_bar)^2, u_bar = 0, U not printed | U = [0, phi_max]^6 (phi_max = 3.49 rad, the dataset's assumption), J raw rad^2, standardised by the trainer (mean 24.4, sd 8.9); variants below |
| Inversion | Davis-Yin | DYS 100 + 2 Newton (+ the IFT step) |

## Sec. V, planar 3-DoF

Training: 5880 steps, 139-221 s per 3-epoch run; 10-epoch probe 504 s; full evaluation 888 s. Validation on 20k held-out samples.

lambda sweep (relu, mu = 0.1, 3 epochs). Task error at the Fig. 3 targets is |p_x(G^-1([y;0])) - y|:

| lambda | val task RMSE [m] | val aux MSE | err at y = 1.00 / 1.95 / 2.90 [mm] | J(q*) - J*(y) at the three targets |
| --- | --- | --- | --- | --- |
| 1 | 0.207 | 0.091 | 21 / 121 / 37 | 0.009 / 0.285 / -0.470 (misses y) |
| 0.1 | 0.050 | 0.076 | 10 / 3 / 42 | (probe) 0.01 / 0.01 / -0.55 |
| **0.01** | **0.0205** | 0.199 | **2.7 / 3.1 / 3.7** | **0.013 / 0.034 / 0.103** |
| 0.001 | 0.011 | 0.390 | 5 / 7 / 13 | 0.02 / 0.10 / 0.39; true-J minimum off z = 0 |
| no head | 0.0074 | - | - | - |

lambda = 1 for 10 epochs plateaus at 0.164 m (mirror convention). The aux loss's large-J samples
(near the joint-limit barrier, J up to 16.7) dominate it; lambda = 0.1 beats lambda = 1 on BOTH
terms.

Fig. 3, fielded relu model (lambda 0.01; `fig3_levelsets_relu.png`, `fig3_configs_relu.png`):

| y | q* = G^-1([y; 0]) | true optimum q (SLSQP, 64 starts) | err [mm] | J(q*) | J* | in [0, pi]^3 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | (1.861, 1.457, 1.596) | (1.958, 1.403, 1.570) | 2.7 | 0.280 | 0.267 | yes |
| 1.95 | (1.368, 0.969, 0.991) | (1.433, 1.075, 0.862) | 3.1 | 1.261 | 1.227 | yes |
| 2.90 | (0.373, 0.275, 0.312) | (0.386, 0.336, 0.275) | 3.7 | 5.738 | 5.635 | yes |

On an 81x81 grid z in [-3, 3]^2 the true J(G^-1([y; z])) is minimised within one grid cell
(0.075) of z = 0 at all three targets. J_hat's level sets are ellipses centred at 0 (axis ratio
~3 at y = 1), not the near-circles of the paper's figure. The paper's colour bar puts J at the
three centres near 0.3 / 1 / 3; our J* are 0.27 / 1.23 / 5.63.

tanh (`fig3_*_tanh.png`): at lr 1e-3 a loss spike near step 4500 (val task 0.093 m); at lr 5e-4
val task 0.064 m, aux 0.78, errors 22 / 37 / 5 mm, gaps 0.06 / 0.10 / 0.33. Same recipe, worse
than relu on every number.

Fig. 4 (`fig4_joint_velocities.png`): y(t) = 1 + 2 smoothstep(t), 200 samples, q*(t) = G^-1([y; 0]),
models relu lambda 0.01. Eq. 17 checked on every chord between samples, in q (bound
scale/mu with scale = pi/2):

| mu | nu learned | max abs(dq)/abs(dy) | bound pi/(2 mu) | min ratio | bound pi/(2 nu) | total variation of qdot | max task err on path | val task RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 10.2 | 2.59 | 31.4 | 0.87 | 0.154 | 20.7 | 4.8 cm | 0.014 m |
| 0.1 | 11.4 | 2.36 | 15.7 | 0.87 | 0.138 | 20.0 | 4.9 cm | 0.021 m |
| 1.5 | 128.2 | 0.996 | 1.047 | 0.97 | 0.012 | 10.1 | 26.6 cm | 0.285 m |

Eq. 17 holds for all three (both sides). mu = 1.5 is smoother (half the total variation), but its
inverse sits AT the bound on the whole path (0.97-0.996 of 1.047), so it cannot supply the gain
dq/dy the arm needs near full stretch (abs(grad p_x) -> 0 as y -> 3) and misses the path by up
to 27 cm; its fit is poor everywhere (val 0.285 m, aux 3.5). A reason it is hard here: J_hat - b
>= abs(z)^2/2 and abs(dz) >= mu abs(dq_hat) along a fibre need J's curvature along the fibre >= ~mu^2
(J_q's is 2 at the centre in q_hat units), so mu = 1.5 is at the edge of representable. Fitting
J x 10 (`--j-scale 0.1`, lambda 1e-4) improved the aux fit (0.75) but not the path. The paper
plots only the target y(t), so its mu = 1.5 tracking is not shown. Our velocity profiles have
the paper's shapes (wiggly at 0.05, smooth at 1.5) with peaks ~2x the paper's, and qdot_2,
qdot_3 have the opposite sign (the paper likely measures the relative joints as pi minus ours,
or plots normalised units; not resolved).

Inversion residual max / mean of abs(G(x) - [y; 0]) over 2000 held-out targets:

| | relu 0.1 f64 | relu f32 | tanh f64 | tanh f32 | mu 0.05 f64 | mu 1.5 f64 |
| --- | --- | --- | --- | --- | --- | --- |
| DYS 10 | 1.3 / 0.37 | same | 2.2 / 0.66 | same | 1.7 / 0.43 | 2.6 / 1.0 |
| DYS 50 | 0.14 / 3.1e-2 | same | 0.54 / 0.32 | same | 0.39 / 7.6e-2 | 0.72 / 0.22 |
| DYS 100 | 1.6e-2 / 3.2e-3 | same | 0.18 / 0.13 | same | 8.1e-2 / 1.6e-2 | 0.22 / 6.5e-2 |
| DYS 200 | 1.7e-3 / 6.6e-5 | same | 6.1e-2 / 2.7e-2 | same | 6.3e-3 / 1.2e-3 | 2.0e-2 / 6.6e-3 |
| DYS 100 + IFT step | 1.7e-3 / 4.3e-5 | same | 2.2e-2 / 1.1e-2 | same | 1.1e-2 / 4.2e-4 | 8.8e-2 / 3.2e-3 |
| + 1 Newton | 6.0e-6 / 1.1e-8 | 6.1e-6 / 2.1e-7 | 2.9e-4 / 6.9e-5 | same | 1.1e-3 / 1.7e-6 | 2.1e-3 / 9.1e-6 |
| + 2 Newton | **3.3e-15 / 3.1e-16** | **1.9e-6 / 1.8e-7** | 5.0e-8 / 7.7e-9 | 1.9e-6 / 6.2e-7 | 3.1e-15 / 3.5e-16 | 2.5e-6 / 1.2e-9 |

DYS 100 alone is NOT converged on trained models (alpha = mu/gam shrinks as nu grows to 10-130);
the two Newton steps are what reach machine precision (float32: ~2e-6).

## Sec. VII, planar HSA soft arm

Data `hsa_n2000000_seed0.npz`: 2e6 samples, all converged; p_x in [-0.41, 0.41] m,
p_y in [0.06, 0.55] m, theta in [-2.53, 2.44] rad. Training 11880 steps, 128-135 s.
Evaluation: 2000 held-out targets, u* = G^-1([y; 0]), forward map SoRoMoX equilibrium.

| HSA run (relu, 3 epochs) | val task RMSE (std units) | val aux MSE [rad^4] |
| --- | --- | --- |
| lambda 1, no clip | 0.106 (spike, lr -> 2e-7) | 0.156 |
| lambda 1, clip 1 | 0.043 | 0.025 |
| **lambda 0.1, clip 1** | **0.025** (p_x 4.5 mm, p_y 1.2 mm, theta 6 mrad) | 0.147 |
| lambda 0.01, clip 1 | 0.018 | 0.93 |

Table II, fielded run (lambda 0.1, clip 1, U = [0, phi_max]^6, u_bar = 0), mean (min-max):

| | Paper LOInK | Ours |
| --- | --- | --- |
| Time | 2.66 ms (2.66-2.67), RTX 4090 | 0.21 ms per target amortised over a batch of 2000 (0.42 s); 34 ms single target; CPU, 6 threads, float32, DYS 100 + 2 Newton |
| EE error at z = 0 | 0.44 mm (0.0022-35.0) | **159 mm (0.62-445)**; u* inside U for 1 of 2000 targets, mean 2.6 rad outside |
| EE error at the sample's own latent z_i = G_2(x_i) | - | 2.29 mm (0.024-12.3), median 1.88 |
| Cost abs(u*)^2 | 0.17 (0.0003-0.59), units unstated | 28.4 rad^2 (9.3-60.0); normalised by phi_max: 2.33 (0.76-4.93); dataset samples 24.6 rad^2 |
| Ratio abs(u*)^2 / reference | ~1 against an optimiser (Fig. 9) | 1.28 (0.60-9.50) against the generating sample (no optimiser here); 1.61 (0.68-7.26) against the cheapest training sample within 2 mm / 0.02 rad |
| Fig. 10, abs(u)^2/abs(u*)^2 at c = 0.1 / 0.2 / 0.3 / 0.4 | all >= ~0.98, spreading to ~1.6 at c = 0.4 | mean 1.06 / 1.11 / 1.17 / 1.23, but 41 / 38 / 35 / 33 % of draws are CHEAPER than z = 0 |

Mechanism: the head's offset b_hat(y) = J_hat at z = 0 is -5.3 rad^2 on average (as low as -115),
below every feasible cost: the fitted quadratic's centre is extrapolated outside the data. With
U = [0, phi_max]^6 and u_bar = 0 the minimiser of abs(u)^2 on a task fibre lies on the boundary of
U, where z = 0 cannot sit, and outside U G_1 is unfitted. Variants (same recipe):

| Variant | u* in U | EE all [mm] | EE, u* in U [mm] | own-latent EE [mm] | b_hat mean [rad^2] |
| --- | --- | --- | --- | --- | --- |
| u_bar = 0, U = [0, phi_max] (fielded) | 0.05 % | 159 | - | 2.29 | -5.3 |
| same, J raw (not standardised), lambda 1e-3 | 0 % | 165 | - | 2.46 | -0.28 |
| u_bar = phi_max/2 (centre of U) | 25.6 % | 88 | 6.9 | 3.33 | -1.25 |
| U = [-phi_max, phi_max], u_bar = 0 (new 2e6 dataset, 429 s) | 17.5 % | 131 | 7.5 | 6.71 (theta wraps at +-pi) | -7.1 |

Figures: `hsa/fig8_ee_error_<run>.png`, `hsa/fig9_cost_ratio_<run>.png`,
`hsa/fig10_sampling_<run>.png`. The IKFlow row of Table II is pending.

## Verdicts

- **Fig. 3: reproduced** (relu, lambda 0.01): z = 0 solutions reach the targets to 3-4 mm, in
  limits, elbow up, within 0.01-0.10 of the true constrained optimum, and match the paper's drawn
  configurations; the true cost over the latent is minimised at z = 0. Level sets are ellipses,
  not near-circles. With lambda = 1 the target misses are 2-12 cm.
- **Fig. 4: reproduced in kind**: eq. 17 holds on every chord for all three mu; mu = 1.5 halves
  the velocity variation, but at mu = 1.5 the model sits on the inverse-Lipschitz bound and misses
  the path by up to 27 cm (the paper shows only the target).
- **Table II: NOT reproduced**: EE error 159 mm v 0.44 mm, because z = 0 decodes outside U in
  ~all targets; the fit itself (own-latent EE 2.3 mm) is ~5x the paper's error; cost ratio > 1
  against random samples. Inference cost per target is not comparable (different device and batching).
- The port: no bug found; inversion to 3e-15 (float64) / 2e-6 (float32) on every trained model.

## Settings to carry into the chart campaign (measured here)

lambda well below 1 (0.01-0.1 with the costs used here); gradient clipping 1.0 with the plateau
scheduler; DYS 100 + 2 Newton (DYS 100 alone leaves 1e-2 to 2e-1); relu over tanh at this budget.
A cost whose per-fibre minimiser lies on the boundary of the training domain (the HSA case) puts
z = 0 outside the domain; the barrier cost (planar) does not.
