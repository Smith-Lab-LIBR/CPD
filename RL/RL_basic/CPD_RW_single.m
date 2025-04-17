% get probabilities of true action for each trial 

function model_output = CPD_RW_single(params, trials, decay_type, settings)
param_names = fieldnames(params);
% parameters assignment
for k = 1:length(param_names)
    param_name = param_names{k};
    param_value = params.(param_name);
    fprintf('%s: %f \n',param_name, param_value);

end
%rng(1)
learning_rate = params.reward_lr;

inverse_temp = params.inverse_temp;
reward_prior = params.reward_prior;
%reward_prior = 0.2843;
if isfield(params, 'decay')
    decay_rate = params.decay;
end

% loop over each trial 
choice_rewards = [reward_prior, reward_prior, reward_prior];
choices = trials;
% retrieve true action/s and results 
for trial = 1:length(trials)
    current_trial = trials{trial};
    if height(current_trial) > 2
        true_actions = current_trial(2:3,2);
   else
        true_actions = current_trial(2,2);
    end
    trial_length = height(true_actions);
    correct_choices = current_trial(1,3);

    if strcmp(decay_type, "basic") && trial > 1
        decay_rate = params.decay;
        choice_rewards = rl_decay(choice_rewards, reward_prior, true_action + 1, decay_rate);
    end
    time_points = choices{trial};
    for t = 1:min(trial_length, 3)
        true_action = true_actions(t, 1).response;
         if ~isempty(correct_choices)
             correct_choice = correct_choices.result;
         end    
        reward_probabilities = softmax_rows(choice_rewards*inverse_temp);
        if t == 1
            outcome = zeros(1,3);

            %% simulate actions
            action_probabilities =reward_probabilities;
            action_probs{trial}(t,:) = action_probabilities;
            u = rand(1, 1);
            choice = find(cumsum(action_probabilities) >= u, 1);
            choice = choice - 1;
            
            time_points.response(t+1) = choice;
            if settings.sim
                true_action = choice;
            end
            
            if t == trial_length
                outcome = outcome -1;
                outcome(true_action + 1) = 1;
                prediction_error = learning_rate*(outcome - choice_rewards);
                choice_rewards = choice_rewards + prediction_error;
            end

        else
            if ~isempty(correct_choice)
                if settings.sim
                    previous_result_idx = choice + 1;
                else
                    previous_result_idx = true_actions(t-1, 1).response + 1;
                end
                outcome = zeros(1, 3);
                outcome = outcome - 1; 
                outcome(correct_choice + 1) = 1; 
                reward_probabilities(:,previous_result_idx) = exp(-16);
                row_sums = sum(reward_probabilities, 2);
                reward_probabilities = bsxfun(@rdivide, reward_probabilities, row_sums);
                action_probabilities = sum(reward_probabilities, 1);
                action_probs{trial}(t,:) = action_probabilities;
                
                u = rand(1, 1);
                choice = find(cumsum(action_probabilities) >= u, 1);
                choice = choice - 1;
                
                time_points.response(t+1) = choice;

                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;

                prediction_error = learning_rate * (outcome(:, columnIndices) - choice_rewards(:, columnIndices));
                choice_rewards(:, columnIndices) = choice_rewards(:, columnIndices) + prediction_error;
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
model_output.action_probs = action_probs;
model_output.simmed_choices = choices;
end



%% functions 
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
