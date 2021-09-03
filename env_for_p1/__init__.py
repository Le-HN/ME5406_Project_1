from gym.envs.registration import register

register(
    id='lirobot-v0',
    entry_point='env_for_p1.envs:LiRobot',
)