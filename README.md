Use adversarial policy/heuristic to improve policy
- every N episodes, do a rollout of the policy B and heuristic A
- compare rewards to go at the end of episodes
- if R_A > R_B improve the rewards of every reward of the episodes in the buffer else diminish


- When cumulative_reward_b > cumulative_reward_a: The rewards for Policy B's transitions are scaled down, reflecting better performance relative to Policy A.
- When cumulative_reward_b < cumulative_reward_a: The rewards for Policy B's transitions are scaled up, reflecting poorer performance relative to Policy A.
- When cumulative_reward_b = cumulative_reward_a: The rewards for Policy B's transitions remain unchanged, reflecting equal performance.
- When cumulative_reward_a = 0: The rewards for Policy B's transitions remain unchanged to avoid division by zero.

Goal is to increase probability of selecting a better action than the baseline by giving a bonus to the transitions that led to the better result. Acting on rewards have several reasons:
- Can't modify logits in continous domain actions spaces as in discrete action spaces
- Bonus given to every reward as credit assignment problem doesn't enable us to determine which actions led to the increase compared to baseline


TODO: compare sequence of actions and scale bonus of each reward based on the difference with the action taken by the adversarial baseline.