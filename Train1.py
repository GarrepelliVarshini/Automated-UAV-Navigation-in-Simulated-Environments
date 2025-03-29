import csv
import gym
from UAV_Autonomous_navigation import AirSimDroneEnv, SACAgent
import numpy as np
import os

# Initialize environment and agent
env = AirSimDroneEnv()
agent = SACAgent(state_dim=13, action_dim=3)
model_path = "sac_model_final.pth"
if os.path.exists(model_path):
    print("🔄 Loading existing model:", model_path)
    agent.load(model_path)
else:
    print("🚀 No existing model found. Training from scratch.")

num_episodes = 50
batch_size = 64  # Batch size for training
train_start = 1000  # Start training after 1000 samples

# Create CSV file and write header
csv_filename = "training_data.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Total_Reward", "Collisions"])

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    collisions = 0  # Track collisions per episode

    for t in range(500):
        action = agent.select_action(state)  # Select action
        next_state, reward, done, info = env.step(action)  # Step in environment

        # Store experience in replay buffer
        agent.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state  # Update state
        total_reward += reward

        # Get total collisions from the environment info (if available)
        collisions = info.get("collision_count", 0)


        # Start training after collecting enough experiences
        if len(agent.replay_buffer) > train_start:
            agent.train(batch_size)

        if done:
            break

    # Save results to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode + 1, total_reward, collisions])

    # Print episode progress
    print(f"Episode {episode+1}, Total Reward: {total_reward}, Collisions: {collisions}")

    # Save model periodically
    if (episode + 1) % 50 == 0:
        agent.save(f"sac_model_{episode+1}.pth")

# Final save
agent.save("sac_model_final.pth")
