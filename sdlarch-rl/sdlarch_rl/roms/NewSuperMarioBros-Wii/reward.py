def reward(previous_state: dict, state: dict, action: list) -> dict[float, bool]:
    """
    Calculate the reward based on the current and next state.
    
    Args:
        previous_state (dict): The oldest state of the environment.
        state (dict): The current state of the environment.
    
    Returns:
        tuple: A tuple containing the float reward and a boolean indicating if the episode is done.
    """
    reward = 0.0
    done = False
    
    if previous_state is None or state is None:
        return reward, done

    max_speed = 19
    max_x = 6681.0

    x = state.get("x", 0)
    lives = state.get("lives", 0)
    c_time = state.get("time", 0) # assuming time is in seconds, if it's in frames, you might want to convert it to seconds by dividing by the frame rate (e.g., 60)
    
    old_x = previous_state.get("x", 0)
    old_lives = previous_state.get("lives", 0)
    old_c_time = previous_state.get("time", 0)

    # 1. existential Reward (Survival)
    if c_time < old_c_time:
        reward -= 0.01 

    # 2. reward per movement (Progress)
    delta_x = x - old_x
    
    if delta_x > 0:
        # reward proportional to the distance covered
        reward += 0.05 * (delta_x / max_speed)
    elif delta_x < 0:
        # penality for moving backwards
        reward -= 0.01 
    elif delta_x == 0:
        # penality for getting stuck on a wall or not moving
        reward -= 0.02

    # 3. penalty for losing a life (Death)
    if lives < old_lives:
        reward -= 5.0
        done = True

    # 4. The biggest reward for reaching the end of the level (Goal)
    if x >= max_x:
        # Bonus for reaching the end of the level
        reward += 10.0 
        
        # bonus for finishing faster (the less time, the bigger the bonus)
        time_bonus = c_time / 50.0 
        reward += time_bonus
        
        done = True


    return reward, done