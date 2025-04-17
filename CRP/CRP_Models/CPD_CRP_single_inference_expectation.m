function model_output = CPD_CRP_single_inference_expectation(params, trials, decay_type, settings)

rng(1)
param_names = fieldnames(params);
for k = 1:length(param_names)
    param_name = param_names{k};
    param_value = params.(param_name);
    fprintf('%s: %f \n',param_name, param_value);

end

learning_rate = params.reward_lr;
inverse_temp = params.inverse_temp;
reward_prior = params.reward_prior;
latent_lr = params.latent_lr;
alpha = params.alpha;
if isfield(params, 'decay')
    decay_rate = params.decay;
end
if isfield(params, 'forget_threshold')
    forget_threshold = params.forget_threshold;
else
    forget_threshold = 0;
end

% loop over each trial
latent_states_distribution = [1];
latent_state_rewards = [reward_prior, reward_prior, reward_prior];
latent_state_counts = [1];
t_latent_state_counts = 1;
temporal_mass = zeros(1, 1);
temporal_mass(1,1) = 1;
timestep = 1;
choices = trials;
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
    
    if strcmp(decay_type, "basic")
        decay_rate = params.decay;
        [latent_state_counts, latent_state_rewards] = crp_basic_decay(latent_state_counts, latent_state_rewards, decay_rate, forget_threshold);
        t_latent_state_counts = sum(latent_state_counts);
    elseif strcmp(decay_type, "temporal")
        decay_rate = params.decay;
        [latent_state_counts, latent_state_rewards, temporal_mass, total_counts] = crp_temporal_weighting_decay(decay_rate, temporal_mass, latent_state_rewards, forget_threshold);
        t_latent_state_counts = total_counts;
    end
    time_points = choices{trial};
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
            action_probabilities = sum(latent_states_distribution' .*reward_probabilities, 1);
            action_probs{trial}(t, :) = action_probabilities; 
            u = rand(1, 1);
            choice = find(cumsum(action_probabilities) >= u, 1);
            choice = choice - 1;
            
            time_points.response(t+1) = choice;
            if settings.sim
                true_action = choice;
            end
            if trial_length == 1 % QUESTION
                % update belief over latent states
                % contruct prior based on CRP
                new_latent_states_distribution = latent_state_counts/(t_latent_state_counts + alpha);
                new_latent_states_distribution(end+1) = alpha/(t_latent_state_counts + alpha);
                latent_state_rewards(end+1,:) = [reward_prior,reward_prior,reward_prior];
                CRP_likelihoods = softmax_rows(latent_state_rewards);
                post = new_latent_states_distribution.*CRP_likelihoods(:, true_action+1)';
                post = (post+exp(-16))/(sum((post+exp(-16))));
                u = rand(1,1);
                new_CRP_idx = find(cumsum(post) >= u, 1);
                outcome = outcome - 1;
                outcome(true_action + 1) = 1;
                prediction_error = learning_rate*(outcome - latent_state_rewards);
                latent_state_rewards = latent_state_rewards + post' .* prediction_error; % weighted by posterior distribution
                if new_CRP_idx ~= length(new_latent_states_distribution)
                    latent_state_rewards(end, :) = [];
                    post(end) = [];
                    post = (post+exp(-16))/(sum((post+exp(-16))));
                else 
                    latent_states_distribution = new_latent_states_distribution;
                    latent_state_counts(end+1) = 0;
                    
                end

                latent_state_counts = latent_state_counts + latent_lr* post;
                t_latent_state_counts = t_latent_state_counts + latent_lr* 1;
                if timestep > size(temporal_mass, 1)
                    temporal_mass(timestep,1:length(post)) = zeros(1,length(post));
                end
                v = merge_vectors(temporal_mass(timestep,:), latent_lr* post);
                temporal_mass(timestep,1:length(v)) = v; 
                timestep = timestep + 1;
            end
    
        else % this is about the second choice (as the first choice at t = 1 was incorrect)
            if ~isempty(correct_choice)
                
               

                if settings.sim
                    previous_result_idx = choice + 1;
                else
                    previous_result_idx = true_actions(t-1, 1).response + 1;
                end
                outcome = zeros(1,3);   
                outcome = outcome - 1;
                outcome(correct_choice + 1) = 1;
                reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);
                reward_probabilities(:,previous_result_idx) = exp(-16);
                row_sums = sum(reward_probabilities, 2); % Sum along the second dimension (rows)
                reward_probabilities = bsxfun(@rdivide, reward_probabilities, row_sums);

                action_probabilities = sum(latent_states_distribution' .* reward_probabilities, 1);
                action_probs{trial}(t,:) = action_probabilities;
                u = rand(1,1);
                choice = find(cumsum(action_probabilities) >= u, 1);
                choice = choice - 1;
                time_points.response(t+1) = choice;
                
                % CRP prior 
                new_latent_states_distribution = latent_state_counts/(t_latent_state_counts + alpha);
                new_latent_states_distribution(end+1) = alpha/(t_latent_state_counts  + alpha);
                latent_state_rewards(end+1,:) = [reward_prior,reward_prior,reward_prior];

                % update latent_state_distribution
                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;
                CRP_likelihoods = softmax_rows(latent_state_rewards);
                post = new_latent_states_distribution.* CRP_likelihoods(:, correct_choice+1)';
                post = (post+exp(-16))/(sum((post+exp(-16))));
                new_CRP_idx = find(cumsum(post) >= u, 1);

                
               
                if new_CRP_idx ~= length(new_latent_states_distribution)
                    latent_state_rewards(end, :) = [];
                    post(end) = [];
                    post = (post+exp(-16))/(sum((post+exp(-16))));
                else 
                    latent_states_distribution = new_latent_states_distribution;
                    latent_state_counts(end+1) = 0;
                    
                end
                
                prediction_error = learning_rate * (outcome(:, columnIndices) - latent_state_rewards(:,columnIndices));
                % update latent state reward predictions weighted by latent state distribution
                latent_state_rewards(:,columnIndices) = latent_state_rewards(:,columnIndices) + post' .* prediction_error;
                
                latent_state_counts = latent_state_counts + latent_lr* post;
                t_latent_state_counts = t_latent_state_counts + latent_lr* 1;
                if timestep > size(temporal_mass, 1)
                    temporal_mass(timestep,1:length(post)) = zeros(1,length(post));
                end
                v = merge_vectors(temporal_mass(timestep,:), latent_lr* post);
                temporal_mass(timestep,1:length(v)) = v; 
                timestep = timestep + 1;
            end
        end
        if ((settings.sim == true && choice == correct_choice) || t == 2)
            time_points.result(t+1) = (choice == correct_choice);
            break
        end
    end
    time_points = time_points(1:t+1,:);
    choices{trial} = time_points;

end
reward_probabilities = softmax_rows(latent_state_rewards*inverse_temp);
model_output.action_probabilities = action_probs;
model_output.simmed_choices = choices;
model_output.reward_probabilities = reward_probabilities;

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











