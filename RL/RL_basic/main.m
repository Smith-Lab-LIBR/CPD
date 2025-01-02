% get probabilities of true action for each trial 

function action_probs = CPD_RL_model(params, trials, test)
rng(1);
% parameters assignment
choices = [];
% learning_rate = 0.3;
% inverse_temp = 0.5;
% initial_value = 0;
% reward_prior_1 = 0;
% reward_prior_2 = 0;
% reward_prior_3 = 0;

learning_rate = params.reward_lr;
inverse_temp = params.inverse_temp;
reward_prior = params.reward_prior;

% loop over each trial 

choice_rewards = [reward_prior, reward_prior, reward_prior];

% retrieve true action/s and results 
for trial = 1:length(trials)
    current_trial = trials{trial};
    if height(current_trial) > 2
        true_actions = current_trial(2:3,2); % true_actions is actions that the pt chose --NL 
   else
        true_actions = current_trial(2,2);
    end
    trial_length = height(true_actions);
    correct_choices = current_trial(1,3);
    for t = 1:min(trial_length, 3)
        true_action = true_actions(t, 1).response;
         if ~isempty(correct_choices)
             correct_choice = correct_choices.result;
         end    
        reward_probabilities = softmax_rows(choice_rewards*inverse_temp);
        if t == 1
            outcome = zeros(1,3);

            %% simulate actions
            action_probabilities = reward_probabilities;
            action_probs{trial}(t,:) = action_probabilities;
            u = rand(1,1);
            choice = find(cumsum(action_probabilities) >= u, 1);

            choice = choice-1;
            choices{trial}(t, :)= choice;
            
            if trial_length > 1 % first choice being wrong (NL)
               prediction_error = learning_rate*(-1 - (choice_rewards(:, true_action+1)));
               choice_rewards(:, true_action+1) = choice_rewards(:, true_action+1) + prediction_error;
            else % first choice being correct -NL
                outcome = outcome -1;
                outcome(true_action + 1) = 1;
                prediction_error = learning_rate*(outcome - choice_rewards);
                choice_rewards = choice_rewards + prediction_error;
            end

        else % second choice (the first choice was wrong) -NL
            if ~isempty(correct_choice)
                previous_result_idx = true_actions(t-1, 1).response + 1;
                outcome = zeros(1, 3);
                outcome = outcome - 1; 
                outcome(correct_choice + 1) = 1; 
                reward_probabilities(:,previous_result_idx) = exp(-16);
                row_sums = sum(reward_probabilities, 2);
                reward_probabilities = bsxfun(@rdivide, reward_probabilities, row_sums);
                action_probabilities = sum(reward_probabilities, 1);
                action_probs{trial}(t,:) = action_probabilities;
                
                u = rand(1,1);
                choice = find(cumsum(action_probabilities) >= u, 1);
                choice = choice - 1; % match the coding of choices from task
                choices{trial}(t,:) = choice;

                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;

                prediction_error = learning_rate * (outcome(:, columnIndices) - choice_rewards(:, columnIndices)); % only the columns where 'columnIndices' is 'true' are considered in the calcu of the PE
              
                choice_rewards(:, columnIndices) = choice_rewards(:, columnIndices) + prediction_error;
            end
        end
    end
end

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
