# Energy Management of Port Microgrid System Using Deep Reinforcement Learning

## Overview

This repository presents an advanced **Energy Management System (EMS)** utilizing **Deep Reinforcement Learning (DRL)** to optimize energy distribution within a sustainable port microgrid. The EMS integrates renewable energy sources, including photovoltaic (PV) systems, battery storage, and hydrogen systems, to balance energy demands, reduce costs, and enhance efficiency.

The project employs two powerful RL agents, **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**, to evaluate performance under dynamic and real-world energy scenarios. It provides a comprehensive framework for managing microgrid operations, delivering actionable insights into energy optimization and sustainability.

---

## Features

- **Customizable Microgrid Environment**:
  - Includes PV, battery, and hydrogen systems with realistic operational constraints.
  - Models energy tariffs, feed-in profits, and hydrogen production costs.
- **Reinforcement Learning Agents**:
  - **DQN**: Efficient for discrete energy control scenarios.
  - **PPO**: Handles continuous action spaces with stability and precision.
- **Scalable and Flexible**:
  - Supports diverse datasets and configurable energy components.
  - Modular design for easy integration with new models or datasets.
- **Detailed Metrics and Visualizations**:
  - Tracks key indicators like battery state-of-charge (SoC), total energy cost, and reward trends.
  - Produces intuitive plots for insights into agent performance.

---

## Methodology

### 1. Microgrid Simulation
- **Environment**: The `EnergyEnv` class simulates the microgrid, incorporating:
  - PV energy generation.
  - Battery operations (charging/discharging).
  - Hydrogen systems for peak energy demand.
- **Cost Modeling**:
  - Implements tariffs for energy purchase, feed-in, and hydrogen costs.
  - Models real-world constraints such as grid dependency and battery efficiency.

### 2. Reinforcement Learning
- **DQN Agent**:
  - Uses a neural network to approximate Q-values for discrete state-action pairs.
  - Learns through experience replay and temporal difference (TD) updates.
- **PPO Agent**:
  - Combines actor-critic models for continuous control.
  - Employs Generalized Advantage Estimation (GAE) for stable policy updates.

### 3. Training Process
- Train agents on realistic datasets with adjustable episodes, learning rates, and reward strategies.
- Monitor performance through metrics such as rewards, energy costs, and system efficiency.

### 4. Validation and Testing
- Validate trained models against unseen datasets.
- Compare DQN and PPO agents in terms of efficiency, adaptability, and robustness.

---

## Results

### Key Observations
- **DQN**: Offers rapid convergence but is limited to discrete actions, making it less adaptable to complex scenarios.
- **PPO**: Demonstrates superior performance in dynamic environments, efficiently handling continuous control tasks.

### Performance Metrics
- **Energy Allocation**: Optimizes PV, battery, and grid usage to minimize costs.
- **Financial Metrics**: Tracks total energy costs, feed-in profits, and overall savings.
- **State-of-Charge**: Maintains optimal battery levels for sustained operations.

### Visual Insights
- Plots generated include:
  - Energy allocation charts.
  - Reward trends over episodes.
  - Battery SoC over time.

---

## Prerequisites

- **Python (>= 3.8)** with the following libraries:
  - TensorFlow (>= 2.6.0)
  - NumPy
  - pandas
  - Matplotlib
  - Seaborn
  - Weights & Biases (`wandb`)
  - TQDM

---
