"""What a benchmark driver needs under `--scene mobile`, in one place for all four drivers.

The drivers keep their own grid loop; this supplies the mobile pieces -- the argument, the
program classes, the target draw, the compartment table of the new room, and the per-cell
starts under the chosen base-start rule -- so the four drivers differ only in a few lines that
call it, rather than in four copies of it.
"""

import numpy as np

from src.mobile_base.starts import BASE_START_RULES, CellRng, MobileGrid
from src.shelf_regions import MOBILE_SHELF_WELDS, ShelfCompartmentRegions
from src.target_screening import MAX_CONSECUTIVE_REJECTIONS, SCENES

MOBILE_MAX_CONSECUTIVE_REJECTIONS = 2_000_000


def AddMobileArgs(p):
    p.add_argument("--base-start", choices=BASE_START_RULES, default=None,
                   help="--scene mobile only: the base-start rule (src/mobile_base/starts.py). "
                        "uniform = base uniform in its box; facing = ../codebase's target-aware "
                        "facing_axis stand-off with jitter; home = (1.5, 0, 0) every cell. "
                        "Applied identically to every arm of the run.")


def CheckMobileArgs(args):
    mobile = getattr(args, "scene", None) == "mobile"
    if mobile and args.base_start is None:
        raise SystemExit("--scene mobile needs --base-start {uniform,facing,home}")
    if not mobile and args.base_start is not None:
        raise SystemExit("--base-start applies only under --scene mobile")
    if mobile:
        bad = [a for a in args.arms.split(",") if a not in ("learned", "numerical")]
        if bad:
            raise SystemExit(f"--scene mobile: only the learned and joint-space arms have a "
                             f"mobile formulation; refused {bad}")
        if getattr(args, "guess_filter", "none") != "none":
            raise SystemExit("--scene mobile: --guess-filter is an analytic-chart filter")
        ## Acceptance in the mobile room is ~1 in 9000 draws (Panda pose, measured
        ## 2026-11-25) against the hardened scene's 0.2-0.7%: at the fixed-base default of
        ## 50000 a 60-target grid would give up about one time in five. ../codebase's PR2
        ## (1 in 10500) used 2e6; so does this, unless the caller set a value.
        if args.max_target_rejections == MAX_CONSECUTIVE_REJECTIONS:
            args.max_target_rejections = MOBILE_MAX_CONSECUTIVE_REJECTIONS
    return mobile


def MobileTagTokens(args):
    return [f"base{args.base_start}"] if args.scene == "mobile" else []


def MobileClass(robot, task, arm):
    from src.mobile_base.robots import MOBILE_CLASSES
    key = ("screw" if robot.startswith("screw") else "soft" if robot.startswith("soft")
           else robot)
    return MOBILE_CLASSES[(key, task, arm)]


class MobileSetup:
    def __init__(self, robot, sampler, args):
        self.sampler = sampler
        self.rule = args.base_start
        self.seed = args.seed
        wrist = SCENES[(robot, "pose")].wrist_frame
        self.grid = MobileGrid(
            sampler,
            lambda v: sampler.collision_free_constraint_eval.Eval(sampler.GridToPlantQ(v)) < 1,
            wrist)
        self.start_draws = []
        self.Attach(sampler)      # the warm-up solve draws native particles on the sampler

    def Draw(self, rng):
        return self.grid.Draw(rng)

    @staticmethod
    def Regions(shelf_inset):
        return ShelfCompartmentRegions(shelf_inset, welds=MOBILE_SHELF_WELDS)

    def Guesses(self, target_qs, n_guesses):
        """`guesses[i][j]` = cell (i, j)'s `q_init` under the rule, each from its own stream."""
        plant, ctx = self.sampler.plant, self.sampler.plant_context
        guesses = []
        for i, q in enumerate(target_qs):
            plant.SetPositions(ctx, self.sampler.GridToPlantQ(q))
            X_WT = self.sampler.frame.CalcPoseInWorld(ctx)
            row = []
            for j in range(n_guesses):
                g, draws = self.grid.Start(self.rule, X_WT, CellRng(self.seed, i, j, self.rule))
                row.append(g)
                self.start_draws.append(draws)
            guesses.append(row)
        return guesses

    def Attach(self, program):
        """Give a per-cell program the cell's base-start rule (the swarm's native base draw)."""
        program.base_start_rule = self.rule
        program.base_start_grid = self.grid
        return program

    def Metadata(self):
        return dict(mobile_base=True, base_start=self.rule,
                    base_limits=[[float(v) for v in b] for b in self.sampler.BaseLimits()],
                    approach_axis_in_target=[float(v) for v in self.grid.approach_T],
                    median_start_draws=float(np.median(self.start_draws))
                    if self.start_draws else None)
