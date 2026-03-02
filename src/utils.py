import logging
import os
import numpy as np
from pydrake.all import(
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    ApplyVisualizationConfig,
    VisualizationConfig,
    RigidTransform,
    Rgba,
    Sphere,
    Cylinder,
    RotationMatrix,
    AutoDiffXd,
    RollPitchYaw_,
    StartMeshcat,
)

def RepoDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class HiddenPrints:
    def __enter__(self):
        # Silence Python loggers (root and ikflow)
        self._loggers = [logging.getLogger(), logging.getLogger("ikflow")]
        self._saved_levels = {lg: lg.level for lg in self._loggers}
        for lg in self._loggers:
            lg.setLevel(logging.CRITICAL)

        # Save original fds
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        # Redirect both to /dev/null (Python + C/C++ writes)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore fds
        os.dup2(self._stdout_fd, 1)
        os.dup2(self._stderr_fd, 2)
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
        os.close(self._devnull)
        # Restore logger levels
        for lg, lvl in self._saved_levels.items():
            lg.setLevel(lvl)
        return False
    


def BuildEnv(meshcat, directives_file=None):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.1)

    # Load the model directives from the YAML file.
    parser = Parser(plant, scene_graph)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    ProcessModelDirectives(LoadModelDirectives(directives_file), plant, parser)

    plant.Finalize()
    vis_config = VisualizationConfig()
    vis_config.publish_illustration = True
    vis_config.publish_proximity = True
    vis_config.publish_inertia = True
    vis_config.delete_on_initialization_event = True   # Clear old visualizations
    ApplyVisualizationConfig(vis_config, builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram


def DrawSphere(target, meshcat, name="/sphere", radius = 0.03, sphere_color = Rgba(1.0, 0.2, 0.2, 0.7)):
    sphere_position = target.translation()
    sphere_radius = radius
    meshcat.Delete(name)
    meshcat.SetObject(name, Sphere(sphere_radius), sphere_color)
    meshcat.SetTransform(name, RigidTransform(sphere_position))

def DrawAxes(pose, meshcat, name="/axes", length=0.1, radius=0.005, alpha=0.8):
    """
    Draw the xyz axes given a RigidTransform.
    Each axis is drawn as a colored cylinder:
      - X: red
      - Y: green
      - Z: blue
    """
    # Remove any previous axes with this name
    meshcat.Delete(name)
    # Get rotation and translation
    R = pose.rotation().matrix()
    origin = pose.translation()

    colors = [Rgba(1, 0, 0, alpha), Rgba(0, 1, 0, alpha), Rgba(0, 0, 1, alpha)]
    axes = [R[:, 0], R[:, 1], R[:, 2]]
    axis_names = ["x", "y", "z"]

    for i, (axis, color, axis_name) in enumerate(zip(axes, colors, axis_names)):
        # Cylinder is along z by default, so compute transform from z to axis
        # Compute rotation from z to axis
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        if np.allclose(v, 0) and c > 0.999:
            rot = np.eye(3)
        elif np.allclose(v, 0) and c < -0.999:
            rot = np.diag([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
        # Place cylinder so its base is at the origin, then translate along axis
        cyl_pose = RigidTransform(R=RotationMatrix(rot), p=origin + axis * length / 2)
        meshcat.SetObject(f"{name}/{axis_name}", Cylinder(radius, length), color)
        meshcat.SetTransform(f"{name}/{axis_name}", cyl_pose)

def DrawCylinder(A, B, meshcat, name="/cylinder", radius=0.03, color=Rgba(0.2, 0.2, 1.0, 0.7)):
    """
    Draw a cylinder from point A to point B in meshcat using Drake geometry.
    A, B: 3D numpy arrays or lists
    meshcat: MeshcatVisualizer instance
    name: path in meshcat
    radius: cylinder radius
    color: Rgba color
    """
    A = np.asarray(A).reshape(3)
    B = np.asarray(B).reshape(3)
    axis = B - A
    length = np.linalg.norm(axis)
    if length < 1e-8:
        # Degenerate case: don't draw
        return
    # Cylinder in Drake is aligned with z-axis by default
    z_axis = np.array([0, 0, 1])
    axis_dir = axis / length

    # Compute rotation matrix that aligns z_axis to axis_dir
    v = np.cross(z_axis, axis_dir)
    c = np.dot(z_axis, axis_dir)
    if np.allclose(v, 0) and c > 0.999:
        R = np.eye(3)
    elif np.allclose(v, 0) and c < -0.999:
        R = np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
    # The cylinder's center is at the midpoint between A and B
    center = (A + B) / 2
    pose = RigidTransform(RotationMatrix(R), center)
    meshcat.Delete(name)
    meshcat.SetObject(name, Cylinder(radius, length), color)
    meshcat.SetTransform(name, pose)

def extract_xyzrpy(pose):
    """pose is numpy array 4x4"""
    pose = pose.GetAsMatrix4()
    ad = isinstance(pose[0, 0], AutoDiffXd)
    type_used = AutoDiffXd if ad else float
    if ad:
        result = np.full(6, AutoDiffXd(0))
    else:
        result = np.zeros(6)
    result[:3] = pose[:3, 3]
    result[3:] = RollPitchYaw_[type_used](pose[:3, :3]).vector()
    return result


def CalculateError(pose1, pose2):
    """
    Compute the translational distance and rotational angle between two RigidTransforms.
    
    Args:
        pose1: First RigidTransform
        pose2: Second RigidTransform
    
    Returns:
        tuple: (angle_rad, distance) where angle_rad is the rotation angle in radians
               and distance is the Euclidean distance between translations
    """
    # Translational distance
    distance = np.linalg.norm(pose1.translation() - pose2.translation())
    
    # Rotational angle: compute relative rotation
    R_rel = pose1.rotation().inverse().multiply(pose2.rotation())
    # Angle from rotation matrix using angle-axis representation
    # trace(R) = 1 + 2*cos(theta)
    trace = R_rel.matrix().trace()
    angle_rad = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    
    return (angle_rad, distance)


class Mug:
    def __init__(self, middle: RigidTransform = RigidTransform(), height: float = 0.04):
        self.middle = middle
        self.height = height

def GenerateDiagramWithMug(q, program, yaml_file, meshcat):
    program.SetPositions(q)
    target = program.frame.CalcPoseInWorld(program.plant_context)
    translation = target.translation()
    rotation = target.rotation().ToRollPitchYaw().vector() * 180 / np.pi

    mug_str = f"""\n  - add_model:\n      name: mug\n      # file: package://drake/examples/manipulation_station/models/shelves.sdf\n      file: package://combining_kinematics/models/mug/mug_simple_red.urdf\n  - add_weld:\n      parent: world\n      child: mug::mug_body_link\n      X_PC:\n        translation: [{translation[0]}, {translation[1]}, {translation[2]}]\n        rotation: !Rpy {{ deg: [{rotation[0]}, {rotation[1]}, {rotation[2]}] }}
    """
    original_size = os.path.getsize(yaml_file)
    with open(yaml_file, "a+") as f:
        f.write(mug_str)
    meshcat.Delete()
    diagram_with_mug = BuildEnv(meshcat=meshcat, directives_file = yaml_file)
    with open(yaml_file, "r+") as f:
        f.truncate(original_size)
    return diagram_with_mug, Mug(target)





