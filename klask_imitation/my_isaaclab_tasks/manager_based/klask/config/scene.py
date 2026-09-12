import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from my_isaaclab_tasks.manager_based.klask.assets.klask_cfg import KLASK_CFG

from .parameters import KLASK_PARAMS

@configclass
class SceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=KLASK_PARAMS["scene"]["ball"]["radius"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=KLASK_PARAMS["events"]["randomize_ball_mass"][
                    "mass_distribution_params"
                ][0]
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=KLASK_PARAMS["events"]["randomize_ball_material"][
                    "restitution_range"
                ][0],
                static_friction=KLASK_PARAMS["events"]["randomize_ball_material"][
                    "static_friction_range"
                ][0],
                dynamic_friction=KLASK_PARAMS["events"]["randomize_ball_material"][
                    "dynamic_friction_range"
                ][0],
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.032)),
    )

    # ✅ this ensures the default joint positions are valid before validation
    klask = KLASK_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Klask",
        init_state=KLASK_CFG.init_state.replace(
            joint_pos={
                "ground_to_slider_1": -0.1,   # within [-0.21, -0.02]
                "ground_to_slider_2":  0.1,   # within [ 0.02,  0.21]
            }
        ),
    )
