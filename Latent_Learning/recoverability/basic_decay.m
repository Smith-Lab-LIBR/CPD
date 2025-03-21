function [latent_states_distribution, rewards] = basic_decay(latent_states, rewards, decay_rate, forget_threshold, recent_index, max_index)
    latent_states_distribution = latent_states * decay_rate;
    latent_states_distribution = latent_states_distribution + exp(-16);
    latent_states_distribution(recent_index) = latent_states(recent_index);  % No decay
    latent_states_distribution(max_index) = latent_states(max_index);  % No decay
    forgotten_states = latent_states_distribution < forget_threshold;
    trimmed_distribution = latent_states_distribution;
    trimmed_distribution(forgotten_states) = [];
    if size(trimmed_distribution) > 0
        rewards(forgotten_states',:) = [];
        latent_states_distribution(forgotten_states) = [];    
    end
    latent_states_distribution = latent_states_distribution/sum(latent_states_distribution);
    
end