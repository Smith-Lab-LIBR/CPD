% get probabilities of true action for each trial 

function model_output = CPD_RW_single(params, trials, decay_type, settings)
param_names = fieldnames(params);
% parameters assignment
for k = 1:length(param_names)
    param_name = param_names{k};
end
% accepting the dot motion
dot_motion_action_prob = nan(2,290);
dot_motion_model_acc = nan(2,290);
dot_motion_rt_pdf = nan(2,290);
num_irregular_rts = 0;


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

           

            if settings.use_DDM
                patch_choice_prob =  action_probabilities(true_action+1);
                if contains(settings.drift_mapping, 'action_prob')
                    drift = params.drift_baseline + params.drift_mod*(patch_choice_prob - .5);
                else
                    drift = params.drift;
                end
                if contains(settings.bias_mapping, 'action_prob')
                    starting_bias = .5 + params.bias_mod*(patch_choice_prob - .5);
                else
                    starting_bias = params.starting_bias;
                end
                % negative drift and lower bias entail greater probability of
                % accepting dot motion, so we check if the person accepted, then
                % flip the sign if necessary
                if settings.sim
                    [simmed_rt, accepted_dot_motion] = simulate_DDM(drift, params.decision_thresh, params.nondecision_time, starting_bias, 1, .001, realmax);
                    % accepted dot motion
                    if accepted_dot_motion
                       % current_trial.result(t+1) = patch_action == correct_choice; % result column is 1 if accepted correct dot motion
                        time_points.accepted_dot_motion(t+1) = 1;
                    end
                    
                    time_points.accept_reject_rt(t+1) = simmed_rt;
                else
                    if  current_trial.accepted_dot_motion(t+1) 
                        drift = drift * -1;
                        starting_bias = 1 - starting_bias;
                    end
                    
                    % make sure valid trial before factoring into log likelihood
                    if current_trial.accept_reject_rt(t+1) >= settings.min_rt && current_trial.accept_reject_rt(t+1) <= settings.max_rt
                        dot_motion_rt_pdf(t,trial) = wfpt(current_trial.accept_reject_rt(t+1) - params.nondecision_time, drift, params.decision_thresh, starting_bias);
                        dot_motion_action_prob(t,trial) = integral(@(y) wfpt(y,drift,params.decision_thresh,starting_bias),0,settings.max_rt - params.nondecision_time); 
                        dot_motion_model_acc(t,trial) =  dot_motion_action_prob(t,trial) > .5;
                    else 
                        num_irregular_rts = num_irregular_rts + 1;
                    end
                end

            end
            
            if ((t == trial_length && ~settings.sim) || (settings.sim && (choice == correct_choice) && ~settings.use_DDM) || (settings.sim && accepted_dot_motion && settings.use_DDM)) 
                outcome = outcome -1;
                outcome(correct_choice + 1) = 1;
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



                %%% USE DDM to fit/simulate probability of accepting dot motion 
                if settings.use_DDM
                    patch_choice_prob =  action_probabilities(true_action+1);
                    if contains(settings.drift_mapping, 'action_prob')
                        drift = params.drift_baseline + params.drift_mod*(patch_choice_prob - .5);
                    else
                        drift = params.drift;
                    end
                    if contains(settings.bias_mapping, 'action_prob')
                        starting_bias = .5 + params.bias_mod*(patch_choice_prob - .5);
                    else
                        starting_bias = params.starting_bias;
                    end

                    if settings.sim
                        [simmed_rt, accepted_dot_motion] = simulate_DDM(drift, params.decision_thresh, params.nondecision_time, starting_bias, 1, .001, realmax);
                        % accepted dot motion
                        if accepted_dot_motion
                           % current_trial.result(t+1) = patch_action == correct_choice; % result column is 1 if accepted correct dot motion
                            time_points.accepted_dot_motion(t+1) = 1;
                        end
                        
                        time_points.accept_reject_rt(t+1) = simmed_rt;
                    else
                    % negative drift and lower bias entail greater probability of
                    % accepting dot motion, so we check if the person accepted, then
                    % flip the sign if necessary
                        if  current_trial.accepted_dot_motion(t+1) 
                            drift = drift * -1;
                            starting_bias = 1 - starting_bias;
                        end
                        
                        % make sure valid trial before factoring into log likelihood
                        if current_trial.accept_reject_rt(t+1) >= settings.min_rt && current_trial.accept_reject_rt(t+1) <= settings.max_rt
                            dot_motion_rt_pdf(t,trial) = wfpt(current_trial.accept_reject_rt(t+1) - params.nondecision_time, drift, params.decision_thresh, starting_bias);
                            dot_motion_action_prob(t,trial) = integral(@(y) wfpt(y,drift,params.decision_thresh,starting_bias),0,settings.max_rt - params.nondecision_time); 
                            dot_motion_model_acc(t,trial) =  dot_motion_action_prob(t,trial) > .5;
                        else 
                            num_irregular_rts = num_irregular_rts + 1;
                        end
                    end
                end

                % Learn from outcome
                columnIndices = true(1, 3);
                columnIndices(previous_result_idx) = false;

                prediction_error = learning_rate * (outcome(:, columnIndices) - choice_rewards(:, columnIndices));
                choice_rewards(:, columnIndices) = choice_rewards(:, columnIndices) + prediction_error;
            end
        end
        if ((settings.sim && settings.use_DDM && accepted_dot_motion) || t == 2)
            time_points.result(t+1) = (choice == correct_choice);
            break
        elseif((settings.sim && (choice == correct_choice) && ~settings.use_DDM) || t == 2)
            time_points.result(t+1) = (choice == correct_choice);
            break
        end
    end
     time_points = time_points(1:t+1,:);
    choices{trial} = time_points;
end
model_output.action_probs = action_probs;
model_output.simmed_choices = choices;

model_output.patch_action_probs = action_probs;
model_output.dot_motion_action_prob = dot_motion_action_prob;
model_output.dot_motion_model_acc = dot_motion_model_acc;
model_output.dot_motion_rt_pdf = dot_motion_rt_pdf;
model_output.num_irregular_rts = num_irregular_rts;

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
