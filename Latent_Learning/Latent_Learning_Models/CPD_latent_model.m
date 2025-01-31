% testing call
%[action_probs, choices] = CPD_Model(1,1,1,1);

function action_probs = CPD_latent_model(params, trials, test)
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
    latent_learning_rate_new = 0.6;
    inverse_temp = 0.5;
    reward_prior = 0.5;
    reward_prior_1 = params.reward_prior_1;

else
    learning_rate = params.reward_lr;
    latent_learning_rate = params.latent_lr;
    latent_learning_rate_new = params.new_latent_lr;
    inverse_temp = params.inverse_temp;
    reward_prior = params.reward_prior;

end
%%%%%%%%%%%%%%%%%%%
%inverse_temp = 2;
latent_states_distribution = [1]; % starts off with one 'real' latent state
new_latent_states_distribution= [.9,.1]; % at the beginning the 'potential' new latent state starts with a probability of 0.1
latent_state_rewards = [reward_prior, reward_prior, reward_prior];
new_latent_state_rewards = [reward_prior, reward_prior, reward_prior;
reward_prior,  reward_prior,  reward_prior];
lr_coef = length(trials);
lr_coef_max = length(trials);
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
            %action_probabilities = sum(latent_states_distribution' .* reward_probabilities, 1);
             [m, idx] = max(latent_states_distribution);
            action_probabilities = reward_probabilities(idx,:);
            action_probs{trial}(t,:) = action_probabilities;
            u = rand(1,1);
            choice = find(cumsum(action_probabilities) >= u, 1);
            % [m, choice] = max(action_probabilities);
            choice = choice - 1; % match the coding of choices from task
            choices{trial}(t,:) = choice;

            if trial_length > 1
                % update latent_state_distribution
                latent_states_distribution = adjust_latent_distribution(latent_states_distribution, reward_probabilities, true_action, latent_learning_rate,0, 0);
                new_latent_states_distribution = adjust_latent_distribution(new_latent_states_distribution, next_reward_probabilities, true_action,latent_learning_rate, latent_learning_rate_new, 0);
                [maxi ,idx] = max(new_latent_states_distribution);
                prediction_error = learning_rate *c* (-1 - (latent_state_rewards(:,true_action + 1))); % selects all rows of latent_state_rewards but only the column corresponding to the action specified by true_action
                prediction_error_next = learning_rate * (-1 - (new_latent_state_rewards(:,true_action + 1)));
                %prediction_error_next = learning_rate *c* (-1 - (new_latent_state_rewards(1:end-1,true_action + 1)));
                latent_state_rewards(:,true_action + 1) = latent_state_rewards(:,true_action + 1) + latent_states_distribution' .* prediction_error;
                %new_latent_state_rewards(1:end-1,true_action + 1) = new_latent_state_rewards(1:end-1,true_action + 1) + new_latent_states_distribution(1:end-1)' .* prediction_error_next;
                new_latent_state_rewards(:,true_action + 1) = new_latent_state_rewards(:,true_action + 1) + new_latent_states_distribution' .* prediction_error_next;
             
            else
                % update latent_state_distribution
                latent_states_distribution = adjust_latent_distribution(latent_states_distribution, reward_probabilities, true_action, latent_learning_rate,0, 1);
                new_latent_states_distribution = adjust_latent_distribution(new_latent_states_distribution, next_reward_probabilities, true_action, latent_learning_rate,latent_learning_rate_new, 1);
                [maxi ,idx] = max(new_latent_states_distribution);
                outcome = outcome - 1;
                outcome(true_action + 1) = 1;
                %prediction_error_next = learning_rate *c* (outcome  - new_latent_state_rewards(1:end-1,:));
                prediction_error = learning_rate * c*(outcome - latent_state_rewards);
                prediction_error_next = learning_rate * (outcome - new_latent_state_rewards);
    
                % update latent state reward predictions weighted by latent state distribution
                latent_state_rewards = latent_state_rewards + latent_states_distribution' .* prediction_error;
                new_latent_state_rewards = new_latent_state_rewards + new_latent_states_distribution' .* prediction_error_next;
                %new_latent_state_rewards(1:end-1,:) = new_latent_state_rewards(1:end-1,:) + new_latent_states_distribution(1:end-1)' .* prediction_error_next;
             
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
                latent_states_distribution = adjust_latent_distribution(latent_states_distribution, reward_probabilities, result, latent_learning_rate,0, 1);
                new_latent_states_distribution = adjust_latent_distribution(new_latent_states_distribution, next_reward_probabilities, result, latent_learning_rate,latent_learning_rate_new, 1);
                [maxi ,idx] = max(new_latent_states_distribution);
                
                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;
                %outcome(sub_block_result.response + 1) = 1;
                prediction_error = learning_rate * c* (outcome(:,columnIndices) - latent_state_rewards(:,columnIndices));
                %prediction_error_next = learning_rate *c* (outcome(:,columnIndices)  - new_latent_state_rewards(1:end-1,columnIndices));
                prediction_error_next = learning_rate * (outcome(:,columnIndices)  - new_latent_state_rewards(:,columnIndices));
                
                % update latent state reward predictions weighted by latent state distribution
                latent_state_rewards(:,columnIndices) = latent_state_rewards(:,columnIndices) + latent_states_distribution' .* prediction_error;
                new_latent_state_rewards(:,columnIndices) = new_latent_state_rewards(:,columnIndices) + new_latent_states_distribution' .* prediction_error_next;
                %new_latent_state_rewards(1:end-1,columnIndices) = new_latent_state_rewards(1:end-1,columnIndices) + new_latent_states_distribution(1:end-1)' .* prediction_error_next;
            else 
       
            end
          end
       


        %if the prospective latent state is the max, we sub out our current latent state distribution and replace it with the new one
        if idx == length(new_latent_states_distribution)
            latent_states_distribution = new_latent_states_distribution;
            %latent_state_rewards(end+1,:) = new_latent_state_rewards(end,:);%[reward_prior_1, reward_prior_2, reward_prior_3];
            latent_state_rewards = new_latent_state_rewards;
            new_latent_states_distribution(new_latent_states_distribution > 0) = new_latent_states_distribution(new_latent_states_distribution > 0) - .1/length(new_latent_states_distribution(new_latent_states_distribution > 0));

            %add new prospective latent state
            new_latent_state_rewards(end+1,:) = [reward_prior, reward_prior, reward_prior];
            new_latent_states_distribution(end+1) = .1;
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

function latent_state_distribution = adjust_latent_distribution(latent_state_distribution, model, action, lr,lr_new ,last_t)

    if last_t == 1 % if the choice was correct   
        model_evidence = model(:,action + 1);
    else % in the case of a multiple timestep trial, i.e first choice is not correct
        model_evidence = 1 - model(:,action + 1); % how wrong was each model
    end
    % get the maximum model evidence
    [max_me, max_me_idx] = max(model_evidence);
    model_evidence = model_evidence';

    % update probability of the latent state with maximum model evidence.
    % This is essentially taking a max - assuming that the most likely
    % latent state was the correct one
    if max_me_idx == length(model_evidence) && lr_new ~= 0
        delta = lr_new * (1 - latent_state_distribution(max_me_idx));
    else
        delta = lr * (1 - latent_state_distribution(max_me_idx));
    end
    latent_state_distribution(max_me_idx) = latent_state_distribution(max_me_idx) + delta;

    % store max latent state distribution
    max_latent_state = latent_state_distribution(max_me_idx);

    % take it out so we can more easily change the other latent states
    latent_state_distribution(max_me_idx) = [];
    model_evidence(max_me_idx) = [];

    %% remove added probability mass from the other latent states (proportional to their relative model evidence compared to the max model evidence) %%
    

    % get ratios
    
    if length(model_evidence) > 1
        model_evidence = 1 - model_evidence;
    end
    me_sum = sum(model_evidence);
    me_ratios = (model_evidence+exp(-16))/(me_sum+exp(-16));
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