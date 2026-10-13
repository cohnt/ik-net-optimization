"""Cross-Entropy warm-up for the particle solvers (`svgd_warmup = "cem"`).

Forward-only iterations on the penalised merit `F + rho0 * (||h~||^2 + ||max(g~, 0)||^2)`
over the N particles: take the top `elite` fraction, refit a diagonal Gaussian (mean and std
of the elites, the std floored at `STD_FLOOR` and smoothed `0.7 * new + 0.3 * previous`),
redraw N-1 particles from it (particle 0 is kept, always), project onto the true variable
bounds. No gradients (`torch.no_grad`), so every evaluate counts as one `map_forward` and
nothing else. A phase of the population method, A/B tested by the benchmark: a clean toggle.

Nothing here knows what a variable means: the Gaussian is over the arm's own decision vector.
"""

import time

import torch

STD_FLOOR = 1e-3
SMOOTH_NEW = 0.7


def cem_warmup(target, X, rho0, iters, elite_frac, gen, deadline=None):
    """`target` is the solver's `_Target` (evaluate + project); `X [N, n]`.

    Returns `(X, stats)` with `stats = {iters, merit_before, merit_after, clip_distance}`.
    """
    N, n = X.shape
    k = max(2, int(round(float(elite_frac) * N)))
    k = min(k, N)
    std_prev = None
    merit_before = None
    merit_after = None
    done = 0
    clip_total = 0.0
    with torch.no_grad():
        for it in range(int(iters)):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            ev = target.evaluate(X, need_grad=False)
            gplus = torch.clamp(ev.g, min=0.0)
            merit = ev.F + float(rho0) * ((ev.h * ev.h).sum(dim=1) + (gplus * gplus).sum(dim=1))
            merit = torch.nan_to_num(merit, nan=float("inf"), posinf=float("inf"))
            if merit_before is None:
                merit_before = float(merit.min().item())
            idx = torch.topk(-merit, k).indices
            elites = X[idx]
            finite = torch.isfinite(elites).all(dim=1)
            elites = torch.where(finite.unsqueeze(1), elites, X[0].unsqueeze(0).expand_as(elites))
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False) if k > 1 else torch.zeros_like(mean)
            std = torch.clamp(std, min=STD_FLOOR)
            if std_prev is not None:
                std = SMOOTH_NEW * std + (1.0 - SMOOTH_NEW) * std_prev
            std_prev = std
            xi = torch.randn(N, n, generator=gen, dtype=X.dtype, device=X.device)
            fresh = mean.unsqueeze(0) + xi * std.unsqueeze(0)
            fresh, clip = target.project(fresh)
            keep0 = torch.zeros(N, dtype=torch.bool, device=X.device)
            keep0[0] = True
            X = torch.where(keep0.unsqueeze(1), X, fresh)
            clip_total += float(torch.where(keep0, torch.zeros_like(clip), clip).sum().item())
            done = it + 1
        ev = target.evaluate(X, need_grad=False)
        gplus = torch.clamp(ev.g, min=0.0)
        merit = ev.F + float(rho0) * ((ev.h * ev.h).sum(dim=1) + (gplus * gplus).sum(dim=1))
        merit_after = float(torch.nan_to_num(merit, nan=float("inf")).min().item())
    return X, dict(iters=done, merit_before=merit_before, merit_after=merit_after,
                   clip_distance=clip_total)
