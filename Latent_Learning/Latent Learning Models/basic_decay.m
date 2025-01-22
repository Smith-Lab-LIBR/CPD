function [latent_states_distribution, rewards] = basic_decay(latent_states, rewards, decay_rate, forget_threshold)
    latent_states_distribution = latent_states*decay_rate;
    forgotten_states = latent_states_distribution < forget_threshold;
    trimmed_distribution = latent_states_distribution;
    trimmed_distribution(forgotten_states) = [];
    if size(trimmed_distribution) > 0
        rewards(forgotten_states',:) = [];
        latent_states_distribution(forgotten_states) = [];    
    end
    
    latent_states_distribution = latent_states_distribution/sum(latent_states_distribution);
    
end