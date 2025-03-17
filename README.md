# Adversarial Soft Actor-Critic (AdversarialSAC)

This project enhances the Soft Actor-Critic (SAC) algorithm by incorporating adversarial heuristics to improve policy performance in reinforcement learning tasks.
Built upon stable_baselines3.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Detailed Infos](#detailed-infos)

## Overview

The AdversarialSAC algorithm introduces an adversarial component to the standard SAC framework. By periodically comparing the agent's policy against a heuristic or baseline policy, the algorithm adjusts the rewards to favor actions that outperform the baseline. This approach aims to enhance the probability of selecting superior actions by rewarding transitions leading to better outcomes.

## Features

- **Adversarial Heuristic Integration:** Periodically evaluates the agent's policy against a baseline heuristic to adjust rewards based on performance comparisons.
- **Dynamic Reward Scaling:** Modifies rewards during training to emphasize transitions that result in better performance relative to the baseline.
- **Enhanced Policy Improvement:** Aims to increase the likelihood of selecting actions that outperform the baseline policy.

## Detailed Infos

### Adversarial Reward Modification Strategy

The Adversarial Soft Actor-Critic (AdversarialSAC) algorithm enhances policy learning by periodically comparing the agent's performance against an adversarial heuristic policy. The goal is to refine the reward signal, increasing the likelihood of selecting actions that outperform the baseline.

#### Adversarial Rollout and Reward Adjustment

Every **N episodes**, the algorithm performs a rollout using:

- **Policy B (Trained Policy)**: The agent’s learned policy.
- **Heuristic A (Baseline Policy)**: A predefined or adversarial heuristic policy.

At the end of each episode, the cumulative rewards (`R_B` for Policy B and `R_A` for Heuristic A) are compared:

- **If `R_B > R_A`** → Increase the rewards for all transitions in the buffer, reinforcing actions that led to better outcomes.
- **If `R_B < R_A`** → Decrease the rewards for all transitions in the buffer, discouraging actions that underperformed.
- **If `R_B = R_A`** → Rewards remain unchanged, as both policies performed equally.
- **If `R_A = 0`** → Rewards remain unchanged to avoid division errors.

#### Why Modify Rewards Instead of Actions?

- In **continuous action spaces**, modifying logits directly is impractical, unlike in discrete action spaces.
- Assigning a **reward bonus to every transition** helps with the **credit assignment problem**, where it’s difficult to determine which specific actions contributed to a performance increase.

#### Future Improvements

- **Action-Based Scaling**: Instead of uniformly adjusting all rewards, compare sequences of actions between Policy B and Heuristic A. Scale the reward bonus for each transition based on how much the action deviates from the adversarial baseline.