from gymnasium.envs.registration import register


register(
    id="resource-gathering-v1",
    entry_point="resource_gathering_modified.resource_gathering_modified:ResourceGatheringModified",
    max_episode_steps=40,
)
