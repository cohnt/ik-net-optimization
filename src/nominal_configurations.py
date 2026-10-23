"""The rigid arms' joint-centering targets, as module-level constants with no heavy imports.

Each program reads its own through `NominalConfiguration` (`PandaIKProgram.Q_NOMINAL`,
`IiwaIKProgram.Q_NOMINAL`), and the reasons for each value are documented there. They live
here, numpy only, so that code which needs the constant without a program -- the LOInK
trainer's matched training cost (`src.charts.loink.TrainingCost`), which runs in a training
job that never imports pydrake -- reads the same array rather than a copy of it. The screw
arm's lives on its spec (`src.screw_arm.params`, already light); the soft arms' is zeros.
"""

import numpy as np

#: Franka's standard "ready" pose (see `PandaIKProgram.Q_NOMINAL`).
PANDA_Q_NOMINAL = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

#: The iiwa's nonsingular in-limits home (see `IiwaIKProgram.Q_NOMINAL`).
IIWA_Q_NOMINAL = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])
