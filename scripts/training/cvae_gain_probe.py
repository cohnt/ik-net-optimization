"""Sample the trained cVAE decoders with no solver (CPU, float64): inside the allowed region,
then with the c box and z ball scaled up, recording |q|_inf and the dominant direction of q.
Run from the repo root under the local memory scope, one core."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
import src.register_robots  # noqa
from src.charts.loading import LoadChart
torch.set_default_device("cpu")
N = 20000
cases = [("panda", "models/panda/panda__cvae7__step620000.pkl", 4.0),
         ("panda", "models/panda/panda__cvae1__step620000.pkl", 2.5),
         ("iiwa14", "models/iiwa14/iiwa14__cvae1__step620000.pkl", 2.5),
         ("panda", "models/panda/panda__cvaef7__step620000.pkl", 4.0),
         ("panda", "models/panda/panda__cvaef1__step620000.pkl", 2.5),
         ("iiwa14", "models/iiwa14/iiwa14__cvaef1__step620000.pkl", 2.5),
         ("iiwa14", "models/iiwa14/iiwa14__cvaef8__step620000.pkl", 4.33)]
g = torch.Generator().manual_seed(0)
for robot_name, ck, R in cases:
    chart = LoadChart(robot_name, ck)
    m = chart.nn_model.to("cpu", torch.float64).eval()
    rob = chart.robot; K = chart.network_width
    lo = np.array([l for l, _ in rob.actuated_joints_limits]); hi = np.array([h for _, h in rob.actuated_joints_limits])
    q0 = lo + (hi - lo) * np.random.default_rng(0).random((N, len(lo)))
    P = rob.forward_kinematics(torch.as_tensor(q0, dtype=torch.float64))
    P = torch.as_tensor(np.asarray(P.cpu()) if torch.is_tensor(P) else P, dtype=torch.float64)[:, :3]
    def sample(c_scale, z_scale):
        pos = P + c_scale * 0.25 * (2 * torch.rand(N, 3, generator=g, dtype=torch.float64) - 1)
        quat = torch.randn(N, 4, generator=g, dtype=torch.float64); quat /= quat.norm(dim=1, keepdim=True)
        u = torch.randn(N, K, generator=g, dtype=torch.float64); u /= u.norm(dim=1, keepdim=True)
        r = R * torch.rand(N, 1, generator=g, dtype=torch.float64) ** (1 / K)
        with torch.no_grad():
            q, _ = chart.decode(m, z_scale * r * u, torch.cat([pos, quat], 1))
        return q.abs().max(1).values.numpy(), q.numpy()
    qi, _ = sample(1, 1)
    print(f"{ck.split('/')[-1][:-4]} (K={K}, ball {R}): in region |q|_inf p50 {np.median(qi):.2f} "
          f"p99.9 {np.percentile(qi, 99.9):.2f} max {qi.max():.2f}; frac>1000 {np.mean(qi > 1000):.5f}")
    for cs, zs in [(10, 1), (100, 1), (1, 10), (1, 100), (100, 100)]:
        qi, q = sample(cs, zs)
        U = q / np.linalg.norm(q, axis=1, keepdims=True); s = np.linalg.svd(U, compute_uv=False)
        _, _, Vt = np.linalg.svd(U, full_matrices=False)
        print(f"    c box x{cs:<3} z ball x{zs:<3}: |q|_inf p50 {np.median(qi):9.3g} max {qi.max():9.3g}; "
              f"top-direction share {s[0]**2/np.sum(s**2):.3f} {np.round(Vt[0], 2).tolist()}")
