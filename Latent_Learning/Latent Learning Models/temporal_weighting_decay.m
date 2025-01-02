function latent_states_distribution = temporal_weighting_decay(latent_states, decay_rate, temporal_probability_mass)
    decay_factors = decay_rate.^(length(temporal_probability_mass) - (1:num_rows)');
    masses = temporal_probability_mass*decay_factors;
    final_masses = sum(masses,2);
    final_masses = final_masses/sum(final_masses);
    latent_states_distribution = latent_states*decay_rate;
end