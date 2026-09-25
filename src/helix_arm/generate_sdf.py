"""Render a `helix7` rung as an SDFormat model, from `params.py` and nothing else.

SDFORMAT RATHER THAN URDF, for two reasons that both come down to what the format can say.
URDF has no screw joint at all -- its spec lists exactly `planar, floating, revolute,
continuous, prismatic, fixed` -- so Drake reads one only through its own `<drake:joint
type="screw">` extension; SDFormat has a native `<joint type="screw">` with
`<screw_thread_pitch>`, so the model says what it is in the format's own words. And Drake
parses `<drake:collision_filter_group>` from SDFormat but NOT from URDF (there is no such
code in `detail_urdf_parser.cc`), so a URDF model would have to push its filters out into
the scene directives -- which would put half the robot's definition in a file about
furniture, and leave the bare model self-colliding for anyone who loaded it alone.

THE JOINT LIMITS IN THIS FILE ARE NOT READ. Drake's parsers call `ParseJointLimits` only
for revolute and prismatic joints, in BOTH formats, so a screw joint's `<limit>` is parsed
and discarded and the plant reports `[-inf, inf]`. They are emitted anyway, because they
are the human-readable record of what `limits.py` sets at load time, and a test asserts the
two agree. See `limits.py` for what the omission costs if it is not repaired.

COLLISION IS SPHERES, VISUAL IS A CYLINDER. Capsules hang Drake's proximity engine -- the
soft arm measured a 6-body capsule model with zero candidate pairs failing to complete one
`MinimumDistanceLowerBoundConstraint` evaluation in minutes, against microseconds for the
identical sphere model. The sphere union IS this robot's collision geometry, declared
rather than approximating anything, so the collision constraint is exact on the robot as
defined; the cylinder is visual only and Drake never collides it.
"""

import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.helix_arm.params import (COLLISION_CHAIN_GAP, LINKS, SPECS, HelixArmSpec)

#: Effort and velocity limits. Nothing in this project reads them -- there is no dynamics
#: and no actuator -- but SDFormat requires the element, and a blank one is worse than an
#: honest placeholder because it reads as a measured zero.
EFFORT = 200.0
VELOCITY = 2.0


def _inertia_block(link) -> str:
    """A solid cylinder of the link's own radius and length, about its centroid."""
    r, L, m = link.radius, link.length, link.mass
    ixx = iyy = m * (3.0 * r * r + L * L) / 12.0
    izz = 0.5 * m * r * r
    cx, cy, cz = (0.5 * (a + b) for a, b in zip(link.a, link.b))
    return f"""      <inertial>
        <pose>{cx:.6f} {cy:.6f} {cz:.6f} 0 0 0</pose>
        <mass>{m:.6f}</mass>
        <inertia>
          <ixx>{ixx:.9f}</ixx><iyy>{iyy:.9f}</iyy><izz>{izz:.9f}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>"""


def _link_block(link, parent_joint) -> str:
    """One link: its frame, its inertia, one cylinder visual and its collision spheres.

    `parent_joint` is the name of the joint whose child this link is, or None for the root.
    SDFormat's URDF-equivalent idiom is that the child link's frame IS the joint frame, so
    the link's pose is `relative_to` the joint and the joint's pose is `relative_to` the
    parent link. That is what makes the link offsets in `params.py` mean what URDF would
    mean by them, which is what the forward-kinematics test checks.
    """
    pose = "" if parent_joint is None else \
        f'\n      <pose relative_to="{parent_joint}">0 0 0 0 0 0</pose>'
    cx, cy, cz = (0.5 * (a + b) for a, b in zip(link.a, link.b))
    visual = f"""      <visual name="{link.name}_visual">
        <pose>{cx:.6f} {cy:.6f} {cz:.6f} 0 0 0</pose>
        <geometry><cylinder><radius>{link.radius:.6f}</radius>
          <length>{link.length:.6f}</length></cylinder></geometry>
      </visual>"""
    spheres = "\n".join(
        f"""      <collision name="{link.name}_sphere{i}">
        <pose>{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} 0 0 0</pose>
        <geometry><sphere><radius>{link.radius:.6f}</radius></sphere></geometry>
      </collision>"""
        for i, c in enumerate(link.sphere_centres()))
    return (f'    <link name="{link.name}">{pose}\n'
            f"{_inertia_block(link)}\n{visual}\n{spheres}\n    </link>")


def _joint_block(joint) -> str:
    """One joint. A screw joint differs from a revolute one by a single element."""
    x, y, z = joint.origin_xyz
    r, p, yw = joint.origin_rpy
    ax, ay, az = joint.axis
    pitch = ""
    if joint.kind == "screw":
        ## Metres per REVOLUTION, and right-handed, which is what SDFormat 1.10+ means by
        ## <screw_thread_pitch>. Its predecessor <thread_pitch> meant radians per metre AND
        ## the opposite handedness, so the two spellings differ by more than a reciprocal.
        pitch = f"\n      <screw_thread_pitch>{joint.pitch:.9f}</screw_thread_pitch>"
    return f"""    <joint name="{joint.name}" type="{joint.kind}">
      <pose relative_to="{joint.parent}">{x:.6f} {y:.6f} {z:.6f} {r:.6f} {p:.6f} {yw:.6f}</pose>
      <parent>{joint.parent}</parent>
      <child>{joint.child}</child>{pitch}
      <axis>
        <xyz>{ax:.6f} {ay:.6f} {az:.6f}</xyz>
        <limit>
          <lower>{joint.lower:.9f}</lower><upper>{joint.upper:.9f}</upper>
          <effort>{EFFORT}</effort><velocity>{VELOCITY}</velocity>
        </limit>
      </axis>
    </joint>"""


def _filter_blocks() -> str:
    """One group per link, ignoring every link nearer than `COLLISION_CHAIN_GAP` along the
    chain. Generated from the same constant `COLLISION_PAIRS` is, so the screen the dataset
    runs and the filters the solver sees cannot disagree -- if they did, the dataset would
    be describing a different robot from the program."""
    out = []
    for i, link in enumerate(LINKS):
        ## The group is `cfg_<link>` rather than `<link>`: SDFormat requires every child of
        ## a model to have a unique name, and a filter group named after its own link
        ## collides with it -- "Non-unique name[base_link] detected 2 times", at parse time.
        near = [f"cfg_{LINKS[j].name}" for j in range(len(LINKS))
                if j != i and abs(j - i) < COLLISION_CHAIN_GAP]
        ignores = "".join(
            f"\n        <drake:ignored_collision_filter_group>{n}"
            f"</drake:ignored_collision_filter_group>" for n in near)
        out.append(
            f'    <drake:collision_filter_group name="cfg_{link.name}">\n'
            f"        <drake:member>{link.name}</drake:member>{ignores}\n"
            f"    </drake:collision_filter_group>")
    return "\n".join(out)


def RenderSdf(spec: HelixArmSpec) -> str:
    links = {j.child: j.name for j in spec.joints}
    body = "\n".join(_link_block(link, links.get(link.name)) for link in spec.links)
    joints = "\n".join(_joint_block(j) for j in spec.joints)
    return f"""<?xml version="1.0"?>
<!-- GENERATED by src/helix_arm/generate_sdf.py from src/helix_arm/params.py.
     Do not edit: a test asserts this file is byte-exactly what the generator emits.

     {spec.name}: screw pitch {spec.pitch:.3f} m/revolution, {spec.travel:.3f} m of stroke
     on the {spec.screw_joint_names[0] if spec.screw_joint_names else "(none)"} joint.

     The <limit> on a screw joint is DISCARDED by every Drake parser; it is recorded here
     and applied at load time by src/helix_arm/limits.py. -->
<sdf version="1.11">
  <model name="{spec.name}">
{body}
{joints}
{_filter_blocks()}
  </model>
</sdf>
"""


def OutputPath(spec: HelixArmSpec, root: str) -> str:
    return os.path.join(root, "models", spec.name, f"{spec.name}.sdf")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    for spec in SPECS.values():
        path = OutputPath(spec, root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(RenderSdf(spec))
        print(f"wrote {os.path.relpath(path, root)}")


if __name__ == "__main__":
    main()
