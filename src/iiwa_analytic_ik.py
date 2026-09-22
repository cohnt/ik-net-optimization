## The iiwa's joint limits, and nothing else.
##
## This file used to carry `Analytic_IK_7DoF`, a full closed-form S-R-S forward/inverse
## kinematics implementation following Faria et al., "Position-based kinematics for 7-DoF
## serial manipulators with global configuration control, joint limit and singularity
## avoidance".  It charted all eight branches and was never reached: there is no
## `Iiwa14IKProgramAnalytic`, so the iiwa comparison is two-way (learned, numerical) while
## the Panda's is three-way.  Writing that arm is future work or possibly not done at all
## (Thomas, 2026-09-19), so the implementation was removed rather than maintained unreached;
## `git show fa0fb95:src/iiwa_analytic_ik.py` has it in full if that arm is ever written.
##
## What survives is the two arrays that `src/iiwa_program.py` (which re-exports them) and
## `scripts/iiwa/iiwa_mug.py` actually import, for drawing random configurations and for the
## joint-limit constraint rows.  Values are the LBR iiwa 14 R820's.

import numpy as np

iiwa_limits_lower = np.array([
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-2.967060,
	-2.094395,
	-3.054326
])
iiwa_limits_upper = np.array([
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	2.967060,
	2.094395,
	3.054326
])
