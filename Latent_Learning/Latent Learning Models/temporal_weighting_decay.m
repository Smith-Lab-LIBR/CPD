function [latent_states_distribution, reward_probabilities, temporal_probability_mass] = temporal_weighting_decay(decay_rate, temporal_probability_mass, reward_probabilities, forget_threshold)
    num_rows = size(temporal_probability_mass, 1);
    decay_factors = decay_rate.^(size(temporal_probability_mass, 1) - (1:num_rows)');
    masses = temporal_probability_mass.*decay_factors;
    final_masses = sum(masses,1);
    final_masses = final_masses/sum(final_masses);
    forgotten_states = final_masses < forget_threshold;
    trimmed_distribution = final_masses;
    trimmed_distribution(forgotten_states) = [];
    if size(trimmed_distribution) > 0
        reward_probabilities(forgotten_states',:) = [];
        temporal_probability_mass(:,forgotten_states) = [];
        final_masses = trimmed_distribution;
    end
    %end
    latent_states_distribution = final_masses;  
    latent_states_distribution = latent_states_distribution/sum(latent_states_distribution);
end