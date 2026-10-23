# bilipnet_golden.npz

Float64 golden reference from the official PLNet JAX/Flax code (github.com/acfr/PLNet, MIT,
`surrogte_loss/rb20d_solver.py` + `layer.py`), for checking a torch BiLipNet port. Everything is float64
(ints for `n`, `depth`, `units`). 98 keys.

Regenerate (throwaway venv, CPU; never the project `.venv`):

    python3 -m venv <scratch>/venv-jax && <scratch>/venv-jax/bin/pip install "jax[cpu]" flax numpy
    JAX_PLATFORMS=cpu <scratch>/venv-jax/bin/python scripts/loink_replication/make_bilipnet_golden.py \
        --plnet-dir <PLNet clone>/surrogte_loss

Versions: jax 0.12.0.1, jaxlib 0.12.0.1, flax 0.12.10, numpy 2.5.4, Python 3.12.3. The script exec()s the
definitions in rb20d_solver.py (`class MonLipNet` .. `mln_RA`) unmodified, because that file's module-level
code (orbax restore, solves, plots) cannot be imported.

## Model

`BiLipNet(units=[16,16,16,16], tau=10.0, depth=2)`, input dim n=5: uni_0, mon_0, uni_1, mon_1, uni_2;
each MonLipNet has `tau = layer_tau = 10**0.5`, ReLU. Flax init (PRNGKey(0), float32, then cast to float64);
then biases overwritten from `np.random.default_rng(0)` in this order: Unitary `b` k=0,1,2 (N(0,0.1^2));
for mon k=0,1: `b0..b3`, `by` (N(0,0.1^2)), `lognu` = log 2 + N(0,0.1^2); then `x` (64,5), then `w` (64,5), both N(0,1).
`W, a, Fq, fq, Fab{j}, fab{j}` stay at flax init (`a`, `fq`, `fab{j}` = float32 Frobenius norm of W/Fq/Fab{j}).

## Keys

Metadata: `n`, `depth`, `units` [4], `tau`, `layer_tau`, `bounds` = [lipmin, lipmax, tau_total] from `BiLipNet.get_bounds`
(lipmin = mu_0*mu_1, lipmax = nu_0*nu_1).

Raw params (key = flax tree path under `params/`; i.e. `uni_0/W` is `params['uni_0']['W']`, and the tree is
`{'params': {'uni_0','mon_0','uni_1','mon_1','uni_2'}}` at the BiLipNet level, i.e. modules named by the lists in `setup`):
- `uni_{k}/W` [5,5], `uni_{k}/a` [1], `uni_{k}/b` [5], k=0,1,2
- `mon_{k}/Fq` [5,64], `fq` [1], `Fab0` [16,16], `Fab{1,2,3}` [32,16], `fab{j}` [1], `b{j}` [16], `by` [5], `lognu` [1], k=0,1

Derived (from `get_params` / `get_bounds`; the torch side should reproduce these from the raw params):
- `uni_{k}/R` [5,5] = cayley((a/||W||_F) W)
- `mon_{k}/mu` = nu/layer_tau, `nu` = exp(lognu), `gam` = nu - mu (scalars)
- `mon_{k}/S` [64,5] (concatenated per-layer S_k, each [16,5]), `mon_{k}/bh` [64] (concat of b0..b3),
  `mon_{k}/V{1,2,3}` [16,16] (get_params' V list, V{j} couples z_{j-1} -> z_j; there is no V0)

Inputs: `x` [64,5], `w` [64,5].

Forward: `uni_{k}_out` (Unitary k on `x`), `mon_{k}_out` (MonLipNet k on `x`), `G_out` (full net on `x`);
`mon_0_helper_b` [64,64] (`mln_fwd_x2b`), `mon_0_helper_z` [64,64] (`mln_fwd_b2z`), `mon_0_helper_y` [64,5]
(`mln_fwd_xz2y`), equal to `mon_0_out` to 1e-10 (asserted).

Inversion (`invert_bilipnet` in the script = the chaining and per-layer Davis-Yin loop of `mln_back_solve_dys`
for an arbitrary target; alpha = 1.0 * mu/gam per layer, Lambda = 1.0, u0 = 0, x recovered with the LAST `zk`):
- `x_inv_k{1,5,20,100,500}` [64,5]: inverse of G at `w`
- `x_inv_of_Gx_k500`: inverse at `G_out`; `recover_err` = max|x_inv_of_Gx_k500 - x| (2.9e-15)
- `mon_0_inv_k500` [64,5]: single-layer inverse of mon_0 at `w`; `mon_0_bz` [64,64] (`mln_bwd_y2b(mon_0,w)`),
  `mon_0_alpha`; `mon_0_dys_{zh,zk,u}_k{1,2,3}` [64,64]: per-step iterates (`zh = relu(u)` used in step s,
  `zk` returned by `mln_RA` in step s, `u` AFTER step s)
- `mon_0_inv_residual_k500` = max|mln_fwd(mon_0, mon_0_inv_k500) - w| (1.3e-15),
  `G_inv_residual_k500` = max|G(x_inv_k500) - w| (3.1e-15). Convergence of G residual vs K: 1.83 (K=1),
  0.28 (5), 7.5e-3 (20), 2.0e-9 (100), 3.1e-15 (500).
