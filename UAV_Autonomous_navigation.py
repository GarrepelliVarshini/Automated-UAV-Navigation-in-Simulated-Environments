import airsim
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from filterpy.kalman import KalmanFilter
import time
import torch.nn.functional as F
import random

# --- Step 1: Define AirSim Gym Environment ---
class AirSimDroneEnv(gym.Env):
    def __init__(self):
        super(AirSimDroneEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.avoid_direction = None

        self.fixed_altitude = -2.0  
        self.action_space = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)  
        self.observation_space = spaces.Box(low=-100, high=100, shape=(11,), dtype=np.float32)  

        # Kalman Filter for obstacle prediction
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf.P *= 1000  # Initial uncertainty
        self.kf.R *= 5  # Measurement noise
        self.obstacle_velocity = np.array([0, 0])
        self.target_position = self.get_target_position()

    def get_target_position(self):
        target_object = "TargetPoint"
        target_pose = self.client.simGetObjectPose(target_object)
        return np.array([target_pose.position.x_val, target_pose.position.y_val, self.fixed_altitude])
    
    def land_at_target(self):
        """ Lands the drone at the target position """
        print("Landing at target location...")
        self.client.landAsync().join()
        print("Drone has landed successfully!")

    def get_lidar_data(self):
        lidar_data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
        
        if len(lidar_data.point_cloud) < 3:
            return 0.0, 0.0, 0.0  

        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Define regions (front, left, right)
        front_obstacles = points[(points[:, 0] > 0) & (np.abs(points[:, 1]) < 0.5)]  
        left_obstacles = points[points[:, 1] > 0.5]
        right_obstacles = points[points[:, 1] < -0.5]

        # Compute distances
        front_dist = np.min(front_obstacles[:, 0]) if len(front_obstacles) > 0 else 0.0
        left_dist = np.min(left_obstacles[:, 1]) if len(left_obstacles) > 0 else -0.0  # Fix sign for left
        right_dist = np.min(right_obstacles[:, 1]) if len(right_obstacles) > 0 else 0.0  

        print(f"Front: {front_dist}, Left: {left_dist}, Right: {right_dist}")  
        if len(front_obstacles) > 0:
            left_edge = np.min(front_obstacles[:, 1])  # Minimum Y (leftmost point)
            right_edge = np.max(front_obstacles[:, 1])  # Maximum Y (rightmost point)
        else:
            left_edge = 0.0
            right_edge = 0.0

        return front_dist, left_dist, right_dist,left_edge,right_edge


    def predict_obstacle_position(self, new_position):
        self.kf.predict()
        self.kf.update(new_position)
        predicted_position = self.kf.x[:2]
        self.obstacle_velocity = self.kf.x[2:]
        return predicted_position

    def step(self, action):
        drone_state = self.client.getMultirotorState().kinematics_estimated.position
        current_pos = np.array([drone_state.x_val, drone_state.y_val, self.fixed_altitude])
        target_x, target_y = self.target_position[:2]
        vx = float(action[0])  # Convert NumPy value to Python float
        vy = float(action[1])  # Convert NumPy value to Python float
        x_difference = abs(current_pos[0] - target_x)
    
        if x_difference < 2.0:  # If x is close, prioritize moving towards the target
            print(x_difference)
            vy = np.sign(target_y - current_pos[1]) * abs(vy)  # Move towards target y
            vx = np.sign(target_x - current_pos[0]) * abs(vx) 
        
        # Move the drone with the given velocity
        self.client.moveByVelocityAsync(vx, vy, 0, 1).join()
        front_distance, left_free_space, right_free_space,left_edgef,right_edgef = self.get_lidar_data()
        obstacle_position = np.array([current_pos[0] + front_distance, current_pos[1]])
        predicted_obstacle = self.predict_obstacle_position(obstacle_position)
        new_state = np.concatenate([
            current_pos, 
            self.target_position, 
            [front_distance, left_free_space, right_free_space],
            self.obstacle_velocity.flatten(),
            [left_edgef,right_edgef]
        ])

        distance_to_target = np.linalg.norm(self.target_position - current_pos)
        reward = -distance_to_target
        ## **2. Bonus for moving in the correct direction**
        movement_vector = np.array([vx, vy])
        target_direction = self.target_position[:2] - current_pos[:2]
        target_direction /= (np.linalg.norm(target_direction) + 1e-6)  
        alignment_score = np.dot(movement_vector, target_direction)  
        reward += alignment_score * 2  # Reward for moving in the right direction

        ## **3. Extra reward for being within a safe corridor (avoiding unnecessary deviation)**
        if abs(current_pos[1] - target_y) < 2.0:  
            reward += 3.0  # Bonus for staying aligned

        ## **4. Bonus for avoiding obstacles smoothly**
        if front_distance > 5.0:  
            reward += 5.0  # Safe path
        elif front_distance < 2.0:  
            reward -= 10.0  # Penalty for getting too close

        if left_free_space > 0.5 or right_free_space < -0.5:
            reward += 3.0  # Encouraging movement into open space

        ## **5. Reward for keeping momentum and avoiding unnecessary stops**
        if np.linalg.norm(movement_vector) > 0.5:
            reward += 2.0  # Reward for not slowing down too much
        done = distance_to_target < 1.0
        if done:
            self.land_at_target()
        return new_state, reward, done, {}

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.target_position = self.get_target_position()
        drone_state = self.client.getMultirotorState().kinematics_estimated.position
        front_distance, left_free_space, right_free_space,left_edgef,right_edgef = self.get_lidar_data()
        current_pos = np.array([drone_state.x_val, drone_state.y_val, self.fixed_altitude])
        new_state = np.concatenate([
            current_pos, 
            self.target_position, 
            [front_distance, left_free_space, right_free_space],
            self.obstacle_velocity.flatten(),
            [left_edgef,right_edgef]
        ])
        
        return new_state
    
class ExpertPolicy:
    def __init__(self):
        self.moving_avoid_direction = None
        self.moving_avoid_counter = 0  # Cooldown for moving obstacle avoidance
        self.static_avoid_direction = None
        self.static_avoid_counter = 0 

# --- Step 2: Expert Action Function ---
    def expert_action(self,state):
        current_pos = np.array(state[:3])
        target_pos = np.array(state[3:6])
        front_distance = state[6]
        left_free_space = state[7]
        right_free_space = state[8]
        obstacle_velocity = np.array(state[9:11])
        left_edge, right_edge = state[11], state[12]
        

        direction_to_target = target_pos - current_pos
        direction_to_target[2] = 0
        direction_to_target /= np.linalg.norm(direction_to_target) + 1e-6
        distance_to_target = np.linalg.norm(direction_to_target)
        direction_to_target /= (distance_to_target + 1e-6)  # Normalize

        future_obstacle_pos = current_pos[:2] + obstacle_velocity * 0.5
        print(f"Predicted Obstacle Future Position: {future_obstacle_pos}")
        # 🔹 Check if the moving obstacle is in the drone's path
        obstacle_ahead = (future_obstacle_pos[0] > current_pos[0]) and (np.linalg.norm(future_obstacle_pos - current_pos[:2]) < 3.0)

        moving_towards_drone = np.linalg.norm(obstacle_velocity) > 0.2  
        action = np.array([direction_to_target[0], direction_to_target[1], 0.0]) 
        current_y = state[1]
        obstacle_center = (left_edge + right_edge) / 2 
        

        # 🛑 **Moving Obstacle Avoidance**
        if obstacle_ahead and moving_towards_drone:
            print("⚠️ Moving obstacle detected in path! Adjusting route.")
            if self.moving_avoid_direction is None:  # Choose direction only once
                
                if right_free_space > 0.5:
                    self.moving_avoid_direction = "right"
                    print("Moving Right to Avoid Moving Obstacle.")
                elif left_free_space > 0.5:
                    self.moving_avoid_direction = "left"
                    print("Moving Left to Avoid Moving Obstacle.")
                else:
                    self.moving_avoid_direction = "stop"
                    print("No space to move! Slowing down.")

                 # Cooldown to prevent rapid switching

            # ✅ Execute Moving Obstacle Avoidance
            if self.moving_avoid_direction == "right":
                print("movv")
                action = np.array([0, -1, 0])  # Move right
            elif self.moving_avoid_direction == "left":
                print("mov")
                action = np.array([0, 1, 0])  # Move left
            elif self.moving_avoid_direction == "stop":
                action = np.array([-0.5, 0, 0])  # Slow down

        elif front_distance < 5.5:  
            if self.static_avoid_direction is None:  # Only choose once per obstacle
                if current_y < obstacle_center and right_free_space < -45:  
                    self.static_avoid_direction = "right"
                elif left_free_space > 0.5:  
                    self.static_avoid_direction = "left"
                else:
                    self.static_avoid_direction = "stop"

                # ✅ Step 2: Execute the chosen action
            if self.static_avoid_direction == "right":
                action = np.array([0, -1, 0])  # Move right
                print("Avoiding obstacle! Moving Right.")
            elif self.static_avoid_direction == "left":
                action = np.array([0, 1, 0])  # Move left
                print("Avoiding obstacle! Moving Left.")
            elif self.static_avoid_direction == "stop":
                action = np.array([-0.5, 0, 0])  # Slow down
                print("Obstacle ahead & no free space! Slowing down.")

        elif left_free_space < 0.5 :
            print("Obstacle on the left! Moving straight.")
            self.static_avoid_direction = None
            self.moving_avoid_direction = None
            action = np.array([-1, 0, 0])  

        elif right_free_space > -45:
            print("Obstacle on the right! Moving straight")
            self.static_avoid_direction = None
            self.moving_avoid_direction = None
            action = np.array([-1, 0, 0])
        else:
            action = np.array([direction_to_target[0], direction_to_target[1], 0.0]) 
            self.static_avoid_direction = None
            self.moving_avoid_direction = None
            print("target") 

        return action




    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
    
    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=-1))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim=6, action_dim=3, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=10000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=0.001)
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=0.001)
        self.target_entropy = -action_dim
        self.expert_action = ExpertPolicy()

    def select_action(self, state):
        expert_act = np.array(self.expert_action.expert_action(state))  # Expert action
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        sac_action, _ = self.actor.sample(state_tensor)
        sac_action = sac_action.detach().cpu().numpy().flatten() 
        # Blend expert and SAC actions
        action = 0.9* expert_act + 0.1* sac_action
        action[2] = 0.0  # Ensure altitude remains fixed
        return np.clip(action, -1, 1)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return  # Skip if buffer isn't full

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ----- Critic Update -----
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)  # Next action
            next_q1 = self.critic_target1(next_states, next_action)
            next_q2 = self.critic_target2(next_states, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * next_q  # Compute target Q

        # Compute current Q-values
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # Backpropagate critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Actor Update -----
        new_action, log_prob = self.actor.sample(states)  # Sample new action
        q1_new = self.critic1(states, new_action)
        q2_new = self.critic2(states, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        # Backpropagate actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Entropy Temperature Update -----
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update alpha
        self.alpha = self.log_alpha.exp()

        # ----- Target Network Update -----
        for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic_target1_state_dict': self.critic_target1.state_dict(),
            'critic_target2_state_dict': self.critic_target2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.item(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic_target1.load_state_dict(checkpoint['critic_target1_state_dict'])
        self.critic_target2.load_state_dict(checkpoint['critic_target2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        
        print(f"Model loaded from {filename}")