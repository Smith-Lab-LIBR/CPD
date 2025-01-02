function action_probs = CPD_CRP_multi_inference_max(params, trials)
rng(1);
choices = [];

learning_rate = params.reward_lr;
inverse_temp = params.inverse_temp;
reward_prior = params.reward_prior;
alpha = params.alpha;

% loop over each trial
latent_states_distribution = [1];
latent_state_rewards = [reward_prior, reward_prior, reward_prior];
latent_state_counts = [1];
t_latent_state_counts = 1;

% retrieve true action/s and results
for trial = 1:length(trials)
    current_trial = trials{trial};
    if height(current_trial) > 2
        true_actions = current_trial(2:3, 2);

    else
        true_actions = current_trial(2, 2);
    end
    trial_length = height(true_actions);
    correct_choices = current_trial(1, 3);
    
    for t = 1:min(trial_length, 3)
        true_action = true_actions(t, 1).response;
        if ~isempty(correct_choices)
            correct_choice = correct_choices.result;
        end

        reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);
        latent_states_distribution = latent_state_counts/t_latent_state_counts;

     
        if t == 1 % this is the first choice 
            outcome = zeros(1, 3);
            
            %% simulate action

            [m, latent_state_idx] = max(latent_states_distribution); 
            action_probabilities = reward_probabilities(latent_state_idx, :);
            action_probs{trial}(t, :) = action_probabilities; 
            [m, choice] = max(action_probabilities);
            choice = choice - 1;
            choices{trial}(t, :) = choice;

             % contruct prior based on CRP
            new_latent_states_distribution = latent_state_counts/(t_latent_state_counts + alpha);
            new_latent_states_distribution(end+1) = alpha/(t_latent_state_counts + alpha);
            latent_state_rewards(end+1,:) = [reward_prior,reward_prior,reward_prior];

            if trial_length > 1 
                % update belief over latent states
                likelihood = softmax_rows(latent_state_rewards);
                update = 1 - likelihood(:, true_action + 1);
                CRP_likelihoods = update;
                post = new_latent_states_distribution.*CRP_likelihoods';
                post = (post+exp(-16))/(sum((post+exp(-16))));
                u = rand(1,1);
                new_CRP_idx = find(cumsum(post) >= u, 1);
                prediction_error = learning_rate*(-1 - latent_state_rewards(latent_state_idx, true_action + 1));
                latent_state_rewards(latent_state_idx, true_action + 1) = latent_state_rewards(latent_state_idx, true_action + 1) + prediction_error;
                if new_CRP_idx ~= length(new_latent_states_distribution)
                    latent_state_rewards(end, :) = [];
                else 
                    latent_states_distribution = new_latent_states_distribution;
                    latent_state_counts(end+1) = 0;
                end

                latent_state_counts(new_CRP_idx) = latent_state_counts(new_CRP_idx) + 1;
                t_latent_state_counts = t_latent_state_counts + 1;
            
            else 
                CRP_likelihoods = softmax_rows(latent_state_rewards);
                post = new_latent_states_distribution.*CRP_likelihoods(:, true_action + 1)';
                post = (post+exp(-16))/(sum((post+exp(-16))));
                u = rand(1,1);
                new_CRP_idx = find(cumsum(post) >= u, 1);
                outcome = outcome - 1;
                outcome(true_action + 1) = 1;

                prediction_error = learning_rate * (outcome - latent_state_rewards(latent_state_idx,:));
                latent_state_rewards = latent_state_rewards +  prediction_error;
                if new_CRP_idx ~= length(new_latent_states_distribution)
                    %latent_states_distribution(end) = [];
                    latent_state_rewards(end,:) = [];
                else
                    latent_states_distribution = new_latent_states_distribution;
                    latent_state_counts(end+1) = 0;
                    
                end

                latent_state_counts(new_CRP_idx) = latent_state_counts(new_CRP_idx) + 1;
                t_latent_state_counts = t_latent_state_counts + 1;
            end
    
        else % this is about the second choice (as the first choice at t = 1 was incorrect)
            if ~isempty(correct_choice)
                
                
                previous_result_idx = true_actions(t-1, 1).response + 1;
                outcome = zeros(1, 3);
                outcome = outcome - 1;
                outcome(correct_choice + 1) = 1;
                
                [m, latent_state_idx] = max(latent_states_distribution);
                
                reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);

                reward_probabilities(latent_state_idx, previous_result_idx) = exp(-16);
                row_sums = sum(reward_probabilities, 2);
                reward_probabilities = bsxfun(@rdivide, reward_probabilities, row_sums);
                action_probabilities = reward_probabilities(latent_state_idx, :);
                action_probs{trial}(t, :) = action_probabilities; 
                [m, choice] = max(action_probabilities);
                choice = choice - 1;
                choices{trial}(t, :) = choice;
                
                % CRP prior 
                new_latent_states_distribution = latent_state_counts/(t_latent_state_counts + alpha);
                new_latent_states_distribution(end + 1) = alpha/(t_latent_state_counts + alpha);
                latent_state_rewards(end+1, :) = [reward_prior, reward_prior, reward_prior];
                
                % update latent_state_distribution
                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;
                CRP_likelihoods = softmax(latent_state_rewards);
                post = new_latent_states_distribution.*CRP_likelihoods(:, correct_choice + 1)';
                post = (post+exp(-16))/(sum((post+exp(-16))));
                u = rand(1,1);
                new_CRP_idx = find(cumsum(post) >= u, 1);

                prediction_error = learning_rate * (outcome(:, columnIndices) - latent_state_rewards(new_CRP_idx,columnIndices));
                % update latent state reward predictions weighted by latent state distribution
                latent_state_rewards(new_CRP_idx,columnIndices) = latent_state_rewards(new_CRP_idx,columnIndices) + prediction_error;
                
                if new_CRP_idx ~= length(new_latent_states_distribution)
                    latent_state_rewards(end, :) = [];
                else 
                    latent_states_distribution = new_latent_states_distribution;
                    latent_state_counts(end+1) = 0;
                end

                latent_state_counts(new_CRP_idx) = latent_state_counts(new_CRP_idx) + 1;
                t_latent_state_counts = t_latent_state_counts + 1;
            end
        end
    end

end

end


%% functions 


function [c,isnew,pmf] = rand_ddCRP(alpha,A,slope,baserate,c_old,N_causes,t)

if isempty(c_old)
    c = 1; isnew = 1; pmf = 1;
    return;
end

% current time point
t_current = t(end);
% previous time points
t_old = t(1:end-1);

% probability for each cause
pmf = zeros(1,N_causes+1);
for i_cause = 1:N_causes
    deltat = t_current - t_old(c_old==i_cause);
    pmf(i_cause) = A*sum(exp(-slope*deltat)) + baserate;
end
pmf(end) = alpha; % new cause
pmf = pmf/sum(pmf); % normalization

% sample
cmf = cumsum(pmf);
c = find(rand()<=cmf,1,'first');

if c > N_causes
    isnew = 1;
else
    isnew = 0;
end

end

function [c,isnew,pmf] = CRP_basic(alpha,c_old,N_causes,t)

if isempty(c_old)
    c = 1; isnew = 1; pmf = 1;
    return;
end

% current time point
t_current = t(end);
% previous time points
t_old = t(1:end-1);

% probability for each cause
pmf = zeros(1,N_causes+1);
for i_cause = 1:N_causes
    deltat = t_current - t_old(c_old==i_cause);
    pmf(i_cause) = A*sum(exp(-slope*deltat)) + baserate;
end
pmf(end) = alpha; % new cause
pmf = pmf/sum(pmf); % normalization

% sample
cmf = cumsum(pmf);
c = find(rand()<=cmf,1,'first');

if c > N_causes
    isnew = 1;
else
    isnew = 0;
end

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

