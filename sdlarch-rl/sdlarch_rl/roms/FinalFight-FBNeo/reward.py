def reward(previous_state: dict, state: dict, action: list = None):
    """
    Calculate the reward based on the current and next state.
    
    Args:
        previous_state (dict): The oldest state of the environment.
        state (dict): The current state of the environment.
        action (list): The action taken in the current state.
    
    Returns:
        tuple: A tuple containing the reward and a boolean indicating if the episode is done.
    """
    reward = 0.0
    done = False 

    health_diff = state["health"] - previous_state["health"]
    points_diff = state["points"] - previous_state["points"]
    x_diff = state["x"] - previous_state["x"]
    was_in_special = previous_state.get("was_in_special", False)
    is_facing_left = state.get("is_left_side", 0) == 1
    number_enemies = state.get("number_enemies", 0)
    level = state.get("level", 0)

    # 1. penality for time (Encourage faster completion)
    reward -= 0.001

    # 2. penality for losing health (Encourage survival)
    if health_diff < 0:
        reward -= 5.0  
    elif health_diff > 0:
        reward += 5.0  # Reward for gaining health, encouraging actions that lead to healing
    
    # 3. Reward for moving right (Encourage progression)
    if x_diff > 0:
        reward += 0.1 

    # 4. Reward for gaining points (Encourage actions that lead to scoring)
    if points_diff > 0:
        if not was_in_special:
            reward += 3.0  # Bonus for gaining points without spamming special moves, encouraging strategic play
        else:
            reward += 1 # Special moves give less total reward

    # 5. Action was mapped from a wrapper, so we need to check the original action list to determine if the special move buttons were pressed.
    # Assuming action[0] corresponds to the attack button and action[8] corresponds to the jump button
    # 6 -> 16 buttons, but we only have 6 in our action space, so we need to check the correct indices based on the wrapper's mapping.
    if action is not None and action[0]  and action[8]:
        reward -= 3  # Negative reward for spamming special moves without strategy

    # 6. Small reward for facing left (Encourage strategic positioning)
    if action is not None and is_facing_left and action[0] and number_enemies > 0:
        reward += 0.05  # Small reward for facing left when there are enemies, encouraging strategic positioning

    # 7. Lose condition (Encourage survival)
    if state["lives"] < previous_state["lives"]:
        done = True

    # 8. Win condition (Encourage completion of the level)
    if level > 0:
        reward += 10
        done = True

    return reward, done