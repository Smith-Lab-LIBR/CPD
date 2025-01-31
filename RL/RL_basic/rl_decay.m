function rewards = rl_decay(rewards, reward_prior, previous_action, decay_rate)
    rewards(~ismember(1:length(rewards), previous_action)) = rewards(~ismember(1:length(rewards), previous_action)) - decay_rate * (rewards(~ismember(1:length(rewards), previous_action)) - reward_prior);
    test = 1;
end
   