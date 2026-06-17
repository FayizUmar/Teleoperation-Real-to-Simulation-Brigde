"""MuJoCo backend for so101-lite: registers all Gymnasium environments.

Importing this module (or ``so101_lite.envs``) registers every env id below
exactly once. Ids keep the ``MuJoCo*`` prefix used by so101-nexus so existing
env names and datasets stay valid.
"""

import gymnasium

_REGISTERED = False


def _register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    gymnasium.register(
        id="MuJoCoPickLift-v1",
        entry_point="so101_lite.envs.pick_env:PickLiftEnv",
        max_episode_steps=1024,
    )
    gymnasium.register(
        id="MuJoCoPickAndPlace-v1",
        entry_point="so101_lite.envs.pick_and_place:PickAndPlaceEnv",
        max_episode_steps=1024,
    )
    gymnasium.register(
        id="MuJoCoReach-v1",
        entry_point="so101_lite.envs.reach_env:ReachEnv",
        max_episode_steps=512,
    )
    gymnasium.register(
        id="MuJoCoLookAt-v1",
        entry_point="so101_lite.envs.look_at_env:LookAtEnv",
        max_episode_steps=256,
    )
    gymnasium.register(
        id="MuJoCoMove-v1",
        entry_point="so101_lite.envs.move_env:MoveEnv",
        max_episode_steps=256,
    )
    gymnasium.register(
        id="MuJoCoChessBoard-v1",
        entry_point="so101_lite.envs.chess_env:ChessBoardEnv",
        max_episode_steps=1024,
    )


_register()
