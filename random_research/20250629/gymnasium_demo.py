import gymnasium as gym

# Create the environment (render_mode="human" shows a window if supported)
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment
# Note: Gymnasium reset returns (observation, info) instead of just observation
obs, info = env.reset()

done = False

while not done:
    # env.render() is no longer needed when using render_mode="human"

    # Choose a random action
    action = env.action_space.sample()

    # Step the environment
    # Gymnasium step returns: obs, reward, terminated, truncated, info
    obs, reward, terminated, truncated, info = env.step(action)

    # Determine if the episode is over
    done = terminated or truncated

    print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

# Always close the environment when done
env.close()
