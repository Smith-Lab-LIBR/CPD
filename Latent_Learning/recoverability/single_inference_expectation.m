% testing call
%[action_probs, choices] = CPD_Model(1,1,1,1);

function action_probs = single_inference_expectation(params, trials, test, decay_type)
param_names = fieldnames(params);
for k = 1:length(param_names)
    param_name = param_names{k};
    param_value = params.(param_name);
    fprintf('%s: %f \n',param_name, param_value);

end

rng(1);
choices = [];
%block_length = data.block_legnth;
%sub_block_length = data.sub_block_length;
%sub_block_probabilities

%% for testing %%
if test == 1
    data_result = [0,0,0,0,2,2,2,2,2,2,0,0,0,0,1,1,1,1,1,1,0,0,0];
    block_length = 1;
    sub_block_length = length(data_result);
    learning_rate = 0.3;
    latent_learning_rate = 0.7;
    latent_learning_rate_new = 0.;
    inverse_temp = 0.5;
    reward_prior = 0.5;
    reward_prior_1 = params.reward_prior_1;

else
    learning_rate = params.reward_lr;
    %learning_rate = 0.606;
    new_lr = 0.1;
    latent_learning_rate = params.latent_lr;
   % latent_learning_rate = 0.895;
   latent_learning_rate_new = 0.46;
   %latent_learning_rate_new = params.new_latent_lr;
    %latent_learning_rate_new = 1;
    %latent_learning_rate_existing = params.existing_latent_lr;
    inverse_temp = params.inverse_temp;
    %inverse_temp = 1.7;
    reward_prior = params.reward_prior;
    %reward_prior = 0;
    if isfield(params, 'decay')
        decay_rate = 1;%params.decay;
    end
    if isfield(params, 'forget_threshold')
        forget_threshold = params.forget_threshold;
    else
        forget_threshold = 0;
    end

end
%%%%%%%%%%%%%%%%%%%
%inverse_temp = 2;
latent_states_distribution = [1]; % starts off with one 'real' latent state
new_latent_states_distribution= [1-new_lr,new_lr]; % at the beginning the 'potential' new latent state starts with a probability of 0.1
latent_state_rewards = [reward_prior, reward_prior, reward_prior];
new_latent_state_rewards = [reward_prior, reward_prior, reward_prior;
reward_prior,  reward_prior,  reward_prior];
lr_coef = length(trials);
lr_coef_max = length(trials);
temporal_mass = zeros(1, 1);
temporal_mass(1,1) = 1;
new_temporal_mass = zeros(1, 2);
new_temporal_mass(1,1) = .9;
new_temporal_mass(1,2) = .1;
timestep = 1;
for trial = 1:length(trials)
    c = lr_coef/lr_coef_max;
    current_trial = trials{trial};
    if height(current_trial) > 2
        true_actions = current_trial(2:3,2);
    else
        true_actions = current_trial(2,2);
    end
    trial_length = height(true_actions);
    results = current_trial(1,3);

     if trial > 1
        %% Handle Decay %%
        if strcmp(decay_type, "basic")
            decay_rate = params.decay;
            [latent_states_distribution, latent_state_rewards] = basic_decay(latent_states_distribution, latent_state_rewards, decay_rate, forget_threshold, max_evidence, max_evidence);
            [new_latent_states_distribution, new_latent_state_rewards] = basic_decay(new_latent_states_distribution, new_latent_state_rewards, decay_rate, forget_threshold, max_evidence_new, max_evidence_new);
            if length(new_latent_states_distribution) <= length(latent_states_distribution)
                new_latent_state_rewards = latent_state_rewards; 
                new_latent_states_distribution = latent_states_distribution;
                new_state_mass = min(1/(length(new_latent_states_distribution) + 1),0.1);
                %new_latent_states_distribution(new_latent_states_distribution > 0) = new_latent_states_distribution(new_latent_states_distribution > 0) * (1-new_state_mass);
                new_temporal_mass = temporal_mass;
                new_temporal_mass = [new_temporal_mass, zeros(size(new_temporal_mass, 1), 1)];
                %add new prospective latent state
                new_latent_state_rewards(end+1,:) = [reward_prior, reward_prior, reward_prior];
                
                new_latent_states_distribution(end+1) = new_state_mass;
                new_latent_states_distribution = new_latent_states_distribution/sum(new_latent_states_distribution);
            end
        elseif strcmp(decay_type, "temporal")
            decay_rate = params.decay;
            [latent_states_distribution, latent_state_rewards, temporal_mass] = temporal_weighting_decay(decay_rate, temporal_mass, latent_state_rewards, forget_threshold);
            [new_latent_states_distribution, new_latent_state_rewards, new_temporal_mass] = temporal_weighting_decay(decay_rate, new_temporal_mass, new_latent_state_rewards, forget_threshold);
            if length(new_latent_states_distribution) <= length(latent_states_distribution)
                new_latent_state_rewards = latent_state_rewards; 
                new_latent_states_distribution = latent_states_distribution;
                new_state_mass = min(1/(length(new_latent_states_distribution) + 1),0.1);
                new_latent_states_distribution(new_latent_states_distribution > 0) = new_latent_states_distribution(new_latent_states_distribution > 0) * (1-new_state_mass);
                new_temporal_mass = temporal_mass;
                new_temporal_mass = [new_temporal_mass, zeros(size(new_temporal_mass, 1), 1)];
                %add new prospective latent state
                new_latent_state_rewards(end+1,:) = [reward_prior, reward_prior, reward_prior];
                new_latent_states_distribution(end+1) = new_state_mass;
                new_latent_states_distribution = new_latent_states_distribution/sum(new_latent_states_distribution);
            end
        end
    end
    for t = 1:min(trial_length, 3)
        true_action = true_actions(t,1).response;
        if ~isempty(results)
            result = results.result;
        end
        reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);
        next_reward_probabilities = softmax_rows(new_latent_state_rewards*inverse_temp);
 
        if t == 1
            outcome = zeros(1,3);
                
            %% simulate action %%
    
            % sample choice from latent states
            action_probabilities = sum(latent_states_distribution' .* reward_probabilities, 1);
            action_probs{trial}(t,:) = action_probabilities;
            u = rand(1,1);
            choice = find(cumsum(action_probabilities) >= u, 1);
            choice = choice - 1; % match the coding of choices from task
            choices{trial}(t,:) = choice;
            %reward_probabilities = (latent_state_rewards+exp(-16))./sum(latent_state_rewards+exp(-16),2);
            %next_reward_probabilities = (new_latent_state_rewards+exp(-16))./sum(new_latent_state_rewards+exp(-16),2);
            % reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);
            % next_reward_probabilities = softmax_rows(new_latent_state_rewards*inverse_temp);
            reward_probabilities = proportionalNormalization(latent_state_rewards);
            next_reward_probabilities = proportionalNormalization(new_latent_state_rewards);
            if t == trial_length
                % update latent_state_distribution
                [latent_states_distribution, temporal_mass, max_evidence] = adjust_latent_distribution(latent_states_distribution, reward_probabilities, true_action, latent_learning_rate,0, 1,  timestep, temporal_mass, decay_type);
                [new_latent_states_distribution, new_temporal_mass, max_evidence_new] = adjust_latent_distribution(new_latent_states_distribution, next_reward_probabilities, true_action, latent_learning_rate,latent_learning_rate_new, 1, timestep, new_temporal_mass, decay_type);
                [maxi ,idx_new] = max(new_latent_states_distribution);
                [maxi ,idx] = max(latent_states_distribution);
                new_prob = new_latent_states_distribution(end);
                outcome = outcome - 1;
                outcome(true_action + 1) = 1;
                evidence = reward_probabilities(:,result+1);
                evidence_new = next_reward_probabilities(:,result+1);
                prediction_error = learning_rate * c*(outcome - latent_state_rewards);
                prediction_error_next = learning_rate * (outcome - new_latent_state_rewards);
                % 
                % % update latent state reward predictions weighted by latent state distribution
                latent_state_rewards = latent_state_rewards + latent_states_distribution' .* prediction_error;
                new_latent_state_rewards = new_latent_state_rewards + new_latent_states_distribution' .* prediction_error_next;
                 % latent_state_rewards = latent_state_rewards + evidence .* learning_rate*outcome;
                 % new_latent_state_rewards = new_latent_state_rewards + evidence_new .* learning_rate*outcome;
                timestep = timestep + 1;
            end
         
        else      
            if ~ isempty(result) % some entries are weird and dont ever have the correct result. These might need to be discarded. TODO double check this later
                previous_result_idx = true_actions(t-1,1).response + 1; % access the action taken in the previous trial; the value in the first column of the row corresponding to the previous trial index (t-1)
                outcome = zeros(1,3);
                outcome = outcome - 1;
                outcome(result + 1) = 1;
                reward_probabilities(:,previous_result_idx) = exp(-16);
                row_sums = sum(reward_probabilities, 2); % Sum along the second dimension (rows)
                reward_probabilities = bsxfun(@rdivide, reward_probabilities, row_sums);
                next_reward_probabilities(:,previous_result_idx) = exp(-16);
                row_sums = sum(next_reward_probabilities, 2); % Sum along the second dimension (rows)
                next_reward_probabilities = bsxfun(@rdivide, next_reward_probabilities, row_sums);
                action_probabilities = sum(latent_states_distribution' .* reward_probabilities, 1);
                action_probs{trial}(t,:) = action_probabilities;
                u = rand(1,1);
                choice = find(cumsum(action_probabilities) >= u, 1);
                % [m, choice] = max(action_probabilities);
                choice = choice - 1; % match the coding of choices from task
                choices{trial}(t,:) = choice;
                % update latent_state_distribution
                %reward_probabilities = (latent_state_rewards+exp(-16))./sum(latent_state_rewards+exp(-16),2);
                %next_reward_probabilities = (new_latent_state_rewards+exp(-16))./sum(new_latent_state_rewards+exp(-16),2);
                %reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);
                %next_reward_probabilities = softmax_rows(new_latent_state_rewards*inverse_temp);
                reward_probabilities = proportionalNormalization(latent_state_rewards);
                next_reward_probabilities = proportionalNormalization(new_latent_state_rewards);
                [latent_states_distribution, temporal_mass, max_evidence] = adjust_latent_distribution(latent_states_distribution, reward_probabilities, result, latent_learning_rate,0, 1, timestep, temporal_mass, decay_type);
                [new_latent_states_distribution, new_temporal_mass, max_evidence_new]= adjust_latent_distribution(new_latent_states_distribution, next_reward_probabilities, result, latent_learning_rate,latent_learning_rate_new, 1, timestep, new_temporal_mass, decay_type);
                [maxi ,idx_new] = max(new_latent_states_distribution);
                new_prob = new_latent_states_distribution(end);
                [maxi ,idx] = max(latent_states_distribution);
                evidence = reward_probabilities(:,result+1);
                evidence_new = next_reward_probabilities(:,result+1);
                % columnIndices = true(1, 3);
                % columnIndices(previous_result_idx) = false;
                % outcome(sub_block_result.response + 1) = 1;
                % prediction_error = learning_rate * c* (outcome - latent_state_rewards);
                % %prediction_error_next = learning_rate *c* (outcome(:,columnIndices)  - new_latent_state_rewards(1:end-1,columnIndices));
                % prediction_error_next = learning_rate * (outcome- new_latent_state_rewards);
                % 
                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;
                %outcome(sub_block_result.response + 1) = 1;
                prediction_error = learning_rate * c* (outcome(:,columnIndices) - latent_state_rewards(max_evidence,columnIndices));
                %prediction_error_next = learning_rate *c* (outcome(:,columnIndices)  - new_latent_state_rewards(1:end-1,columnIndices));
                prediction_error_next = learning_rate * (outcome(:,columnIndices)  - new_latent_state_rewards(max_evidence_new,columnIndices));
                % % update latent state reward predictions weighted by latent state distribution
                % latent_state_rewards= latent_state_rewards+ latent_states_distribution' .* prediction_error;
                % new_latent_state_rewards = new_latent_state_rewards + new_latent_states_distribution' .* prediction_error_next;
                latent_state_rewards(:,columnIndices) = latent_state_rewards(:,columnIndices) +  prediction_error;
                new_latent_state_rewards(:,columnIndices) = new_latent_state_rewards(:,columnIndices) + prediction_error_next;
                % latent_state_rewards = latent_state_rewards + evidence .* learning_rate*outcome;
                %  new_latent_state_rewards = new_latent_state_rewards + evidence_new .* learning_rate*outcome;
                timestep = timestep + 1;
                %new_latent_state_rewards(1:end-1,columnIndices) = new_latent_state_rewards(1:end-1,columnIndices) + new_latent_states_distribution(1:end-1)' .* prediction_error_next;
            else 
       
            end
          end
       


        %if the prospective latent state is the max, we sub out our current latent state distribution and replace it with the new one
        if t == trial_length && new_prob == max(new_latent_states_distribution)
       %if t == trial_length && new_prob > latent_learning_rate_new% max(new_latent_states_distribution)
           latent_states_distribution = new_latent_states_distribution;
            temporal_mass = new_temporal_mass;
            %latent_states_distribution(end+1) = new_latent_states_distribution(end);
            %latent_states_distribution(1:end-1) = latent_states_distribution(1:end-1) - latent_states_distribution(1:end-1) * (latent_states_distribution(end));
            %latent_state_rewards(end+1,:) = new_latent_state_rewards(end,:);%[reward_prior_1, reward_prior_2, reward_prior_3];
            %new_latent_states_distribution = latent_states_distribution;
            latent_state_rewards = new_latent_state_rewards;
            new_state_mass = min(1/(length(new_latent_states_distribution) + 1),new_lr);
            %new_state_mass = new_lr;
            %new_latent_states_distribution(new_latent_states_distribution > 0) = new_latent_states_distribution(new_latent_states_distribution > 0) * (1-new_state_mass);
            
            new_temporal_mass = [new_temporal_mass, zeros(size(new_temporal_mass, 1), 1)];
            %add new prospective latent state
            new_latent_state_rewards(end+1,:) = [reward_prior, reward_prior, reward_prior];
            new_latent_states_distribution(end+1) = new_state_mass;
            new_temporal_mass(timestep,size(new_temporal_mass, 2)) = new_state_mass;
            new_latent_states_distribution = new_latent_states_distribution/sum(new_latent_states_distribution);
            test = 1;
           % lr_coef = lr_coef_max;
        end
    end
   % lr_coef = lr_coef-1;
end
end


%% Helper functions. Most of these unused for now but kept for posterity! %%

function isBelowThreshold = checkKLDivergence(p, Q, threshold)
  % Calculate KL divergence between p and each row of Q
 % Calculate KL divergence using safer log function
  kldiv = sum(bsxfun(@times, p, log2(max(eps, bsxfun(@rdivide, p, Q)))), 2);

  % Check if any KL divergence is below the threshold
  isBelowThreshold = any(kldiv < threshold);
end

function matrix = softmax_rows(matrix)
    % Subtract the maximum value from each row for numerical stability
    matrix = bsxfun(@minus, matrix, max(matrix, [], 2));
    
    % Calculate the exponent of each element
    exponents = exp(matrix);
    
    % Calculate the sum of exponents for each row
    row_sums = sum(exponents, 2);
    
    % Divide each element by the sum of its row
    matrix = bsxfun(@rdivide, exponents, row_sums);
end

function log_sums_bf = getBayesFactorsAggregation(models, data)
    index = find(data == 1);
    model_evidence = models(:,index);

    bayes_factors = bsxfun(@rdivide, model_evidence, model_evidence.');
    % Create mask to exclude diagonal elements
    mask = logical(~eye(size(bayes_factors)));

    % log sum aggregation of bfs
    log_sums_bf = sum(log(bayes_factors) .* mask, 1);

    % ratio ranks
    log_sums_bf = log_sums_bf + exp(-16);

end

function normArr = proportionalNormalization(arr)
    minVal = min(arr);
    adjusted = arr - minVal; % Shift to make all values positive
    normArr = (adjusted + exp(-16)) ./ sum(adjusted +exp(-16),2); % Scale to sum to 1
end
