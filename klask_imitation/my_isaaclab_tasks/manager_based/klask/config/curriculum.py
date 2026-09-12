import isaaclab.envs.mdp as mdp
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    pass
    # goal_scored = CurriculumTermCfg(
    #     func=mdp.curriculums.modify_reward_weight,
    #     params={
    #         "term_name": "collision_player_ball",
    #         "weight": 0.0,
    #         "num_steps": 16 * 62500,
    #     },
    # )
