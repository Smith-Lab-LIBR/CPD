function [latent_state_distribution, temporal_mass, max_me_idx] = adjust_latent_distribution(latent_state_distribution, model, action, lr, lr_new, last_t, timestep, temporal_mass, decay_type)

    if last_t == 1 % if the choice was correct   
        model_evidence = model(:,action + 1);
    else % in the case of a multiple timestep trial, i.e first choice is not correct
        model_evidence = 1 - model(:,action + 1); % how wrong was each model
    end
    %model_evidence = (model_evidence+exp(-16)/sum(model_evidence+exp(-16)));
    % get the maximum model evidence
    max_me = max(model_evidence);
    max_me_idxs = find(model_evidence == max_me);  % Find all indices where max occurs

% Randomly select one of the max indices
    max_me_idx = max_me_idxs(randi(length(max_me_idxs)));
    model_evidence = model_evidence';
    likelihoods = model_evidence;
    % update probability of the latent state with maximum model evidence.
    % This is essentially taking a max - assuming that the most likely
    % latent state was the correct one
    %if max_me_idx == length(model_evidence) && lr_new ~= 0
    %if lr_new ~= 0

        delta = lr * (1 - latent_state_distribution(max_me_idx));%likelihoods(max_me_idx);
        if delta + latent_state_distribution(max_me_idx) > 1
            delta = 1-latent_state_distribution(max_me_idx);
        end
    %else
        % delta = lr*likelihoods(max_me_idx); %* (1 - latent_state_distribution(max_me_idx));
        % if delta + latent_state_distribution(max_me_idx) > 1
        %     delta = 1-latent_state_distribution(max_me_idx);
        % end

    %end
    
    if strcmp(decay_type,"temporal")
        if timestep <= size(temporal_mass, 1) && max_me_idx <= size(temporal_mass, 2)
            temporal_mass(timestep, max_me_idx) = temporal_mass(timestep, max_me_idx) + delta;
        else
            temporal_mass(timestep, max_me_idx) = delta;
        end
         final_masses = sum(temporal_mass,1);
         latent_state_distribution = final_masses/sum(final_masses);
    else
        latent_state_distribution(max_me_idx) = latent_state_distribution(max_me_idx) + delta;
        max_latent_state = latent_state_distribution(max_me_idx);

        % take it out so we can more easily change the other latent states
        latent_state_distribution(max_me_idx) = [];
        model_evidence(max_me_idx) = [];
    
        %% remove added probability mass from the other latent states (proportional to their relative model evidence compared to the max model evidence) %%
      
        % get ratios    
        if length(model_evidence) > 1
            model_evidence = 1 - model_evidence;
        end
        me_sum = sum(model_evidence + exp(-16));
        me_ratios = (model_evidence+exp(-16))/(me_sum);
        % partition delta (amount added to max latent state) with these ratios
        mass_deltas = delta .* me_ratios;
    
        % subtract partitions
        latent_state_distribution = latent_state_distribution - mass_deltas;
        
        
        % Some numbers might be negative. We need to normalize. To be
        % fancy we could normalize in a 'proportional' way where the mass
        % added back to the negative values to make the 0 is proportionally subtracted
        % from the non-negative values based on their model evidence ratios. I
        % dont think this is really needed though, but worth maybe looking
        % into at a later point. This will need to be 'recursive' as subtracting
        % probability mass might create new negative probabilities
    
        while any(latent_state_distribution < 0)
            
            % get the mass we will have to take away from the non-negative states
            summed_negative_values = sum(latent_state_distribution(latent_state_distribution < 0));
    
            % set the negative states to 0 
            latent_state_distribution(latent_state_distribution < 0) = 0;
         
            % subtract the added mass from non-negative states
            latent_state_distribution(latent_state_distribution > 0) = latent_state_distribution(latent_state_distribution > 0) + summed_negative_values/length(latent_state_distribution(latent_state_distribution > 0));
            test = 1;
        end
        % We have finished working with the 'sub-max' part of the latent state distribtuion add back in the max latent state
        latent_state_distribution = [latent_state_distribution(1:max_me_idx-1), max_latent_state, latent_state_distribution(max_me_idx:end)];
        test = 1;
   end
end