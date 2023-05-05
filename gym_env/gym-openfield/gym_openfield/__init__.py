from gym.envs.registration import register

register(
    id='openfield-v0',
    entry_point='gym_openfield.envs:OpenFieldEnv',
)
