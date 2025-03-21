clear
distribution_9_3_1 = [9/13, 1/13, 3/13];
distribution_9_1_3 = [1/13, 9/13, 3/13];


for i = 1:300
    u = rand(1,1);
    if i < 30 || mod(floor(i/30),2) == 0
        choice = find(cumsum(distribution_9_3_1) >= u, 1);
    else
        choice = find(cumsum(distribution_9_1_3) >= u, 1);
    end
    choices(i,:) = choice;
end

simulation_test(choices)
function [] = simulation_test(choices)
    
    if ispc
        %root = 'C:/';
        root = 'L:/';
    elseif isunix 
        root = '/media/labs/';  
    end
    

    trial_num = 1;
   
    %%%%%%%%%%%%%%
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/']);
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/']);  
        
    outer_fit_list = {@simulated_testing};
    %outer_fit_list = {@CPD_latent_single_inference_expectation, @CPD_latent_single_inference_max};
    %outer_fit_list = {@CPD_latent_single_inference_max};
    %inner_fit_list = {'vanilla', 'basic', 'temporal', 'basic_forget', 'temporal_forget'};
    inner_fit_list = {'vanilla'};
    for i = 1:length(outer_fit_list)
        model_handle = outer_fit_list{i};
       
            for j = 1:length(inner_fit_list)
             if exist('DCM', 'var') && isfield(DCM, 'MDP')
                DCM = rmfield(DCM, 'MDP');
             end
            reward_lr = 0.5;
            latent_lr = 0.5;
            new_latent_lr = 0.5;
            inverse_temp = 1;
            reward_prior = 0;
            decay = 0.9;
            forget_threshold = 0.05;
            params = struct();
            params.reward_lr = reward_lr;
            params.latent_lr = latent_lr;
            params.new_latent_lr = new_latent_lr;
            params.inverse_temp = inverse_temp;
            params.reward_prior = reward_prior;
             if strcmp(inner_fit_list{j}, 'vanilla')
                 decay_type = "";
             elseif strcmp(inner_fit_list{j}, 'basic_forget')
                decay_type = "basic";
             elseif strcmp(inner_fit_list{j}, 'temporal_forget')
                decay_type = "temporal";
             else
                 decay_type = inner_fit_list{j};
             end

           
            
            %% Get simulated choices using fit parameter values
            % params, trials, test, decay_type, is_simulated_trials, simulated_trials 
             if isequal(outer_fit_list{i}, @single_inference_expectation)
                 model_handle = @latent_single_inference_expectation_simulated;
                 DCM.model = @latent_single_inference_expectation_recov;
             elseif isequal(outer_fit_list{i}, @single_inference_max)
                 model_handle = @latent_single_inference_max_simulated;
                 DCM.model = @latent_single_inference_max_recov;
             end
            %[action_probabilities, choices] = model_handle(params, choices, decay_type);

            %% Now just do normal fitting with these simulated choices
            reward_lr = 0.5;
            latent_lr = 0.5;
            new_latent_lr = 0.5;
            inverse_temp = 1;
            %reward_prior = 0;
            decay = 0.9;
            forget_threshold = 0.05;
            DCM.MDP.reward_lr = reward_lr;
            DCM.MDP.latent_lr = latent_lr;
            DCM.MDP.new_latent_lr = new_latent_lr;
            %DCM.MDP.existing_latent_lr = existing_latent_lr;
            DCM.MDP.inverse_temp = inverse_temp;
            %DCM.MDP.reward_prior = reward_prior;
            DCM.model = @simulated_testing;
            if strcmp(inner_fit_list{j}, 'vanilla')
                 DCM.field  = {'inverse_temp' 'new_latent_lr' 'reward_lr' 'latent_lr'}; % Parameter field
                 %file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/ind_mat/%s_individual_%s.mat'], subject_id, func2str(DCM.model));
                 %filename = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/recoverability/2params/%s_individual_%s.csv'], subject_id, func2str(DCM.model));
            elseif strcmp(inner_fit_list{j}, 'basic') || strcmp(inner_fit_list{j}, 'temporal')
                DCM.MDP.decay = decay;
                DCM.field  = {'reward_lr' 'inverse_temp' 'latent_lr' 'new_latent_lr', 'decay'}; % Parameter field
                %DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior' 'new_latent_lr', 'decay'}; % Parameter field
                file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/ind_mat/%s_individual_%s_%s.mat'], subject_id, func2str(DCM.model), decay_type);
                filename = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/recoverability/%s_individual_%s_%s.csv'], subject_id, func2str(DCM.model), decay_type);
            else
                DCM.MDP.decay = decay;
                DCM.MDP.forget_threshold = forget_threshold; 
                DCM.field  = {'reward_lr' 'inverse_temp' 'latent_lr' 'new_latent_lr', 'decay', 'forget_threshold'}; % Parameter field
                file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/ind_mat/%s_individual_%s_%s_forget.mat'], subject_id, func2str(DCM.model), decay_type);
                filename = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/recoverability/%s_individual_%s_%s_forget.csv'], subject_id, func2str(DCM.model), decay_type);
            end
            MDP.trials = choices;   
            DCM.U = MDP.trials;
            DCM.Y = 0;
            DCM.decay_type = decay_type;
            DCM.simulated_choices = choices;
            CPD_fit_output= fit_test(DCM);
            
            % we have the best fit model parameters. Simulate the task one more time to
            % get the average action probability and accuracy with these best-fit
            % parameters
            params = struct();
            if isfield(CPD_fit_output.Ep, 'reward_lr')
                params.reward_lr = 1/(1+exp(-CPD_fit_output.Ep.reward_lr));
            end
            if isfield(CPD_fit_output.Ep, 'inverse_temp')
                params.inverse_temp = exp(CPD_fit_output.Ep.inverse_temp);
            end
            if isfield(CPD_fit_output.Ep, 'reward_prior')
                params.reward_prior = CPD_fit_output.Ep.reward_prior;
            end
            if isfield(CPD_fit_output.Ep, 'latent_lr')
                params.latent_lr = 1/(1+exp(-CPD_fit_output.Ep.latent_lr));
            end
            if isfield(CPD_fit_output.Ep, 'new_latent_lr')
                params.new_latent_lr = 1/(1+exp(-CPD_fit_output.Ep.new_latent_lr));
            end
            if isfield(CPD_fit_output.Ep, 'decay')
                params.decay = 1/(1+exp(-CPD_fit_output.Ep.decay)); 
            end
            if isfield(CPD_fit_output.Ep, 'forget_threshold')
                params.forget_threshold = 1/(1+exp(-CPD_fit_output.Ep.forget_threshold)); 
            end
        
            % rerun model a final time
            L = 0;
            action_probabilities = DCM.model(params, choices, decay_type); 
            count = 0;
            average_accuracy = 0;
            average_action_probability = 0;
            accuracy_count = 0;
            % compare action probabilities returned by the model to actual actions
            % taken by participant (as we do in Loss function in CPD_fit
            %rng(1)
            for t = 1:length(trials)
                trial = choices{t};
   
                first_choice = trial(1,1);
                if height(trial) > 1 
                    second_choice = trial(2,1);       
                   
                    L = L + log(action_probabilities{t}(1,first_choice + 1) + eps)/2;
                    L = L + log(action_probabilities{t}(2,second_choice + 1) + eps)/2;
                    average_action_probability = average_action_probability + action_probabilities{t}(1,first_choice + 1);
                    [maxi, idx_first] = max(action_probabilities{t}(1,:));
                    if idx_first == first_choice +1
                        average_accuracy = average_accuracy + 1;
                    end
                    accuracy_count = accuracy_count +1;
                    count = count + 1;
                    average_action_probability = average_action_probability + action_probabilities{t}(2,second_choice + 1);
                    [maxi, idx_first] = max(action_probabilities{t}(2,:));
                     if idx_first == second_choice +1
                        average_accuracy = average_accuracy + 1;
                     end
                    count = count + 1;
                     accuracy_count = accuracy_count +1;
                else
                    Likelihood = log(action_probabilities{t}(1,first_choice + 1) + eps);
                    L = L + Likelihood;
                    ap = action_probabilities{t}(1,first_choice + 1);
                    [maxi, idx_first] = max(action_probabilities{t}(1,:));
                     if idx_first == first_choice +1
                        average_accuracy = average_accuracy + 1;
                    end
                    average_action_probability = average_action_probability + ap ;
                    count = count + 1;
                     accuracy_count = accuracy_count +1;
            
                end
            end
            
            %These are the final values. 
            action_accuracy = average_action_probability/count;
            accuracy = average_accuracy/accuracy_count;
            
            fprintf('Final LL: %f \n',L)
            fprintf('Final Average choice probability: %f \n',action_accuracy)
            fprintf('Final Average Accuracy: %f \n',accuracy)          
          
            %save(file_name)
            output = struct();
            output.subject = subject_id;
            output.reward_lr = params.reward_lr;
            output.latent_lr = params.latent_lr;
            output.new_latent_lr = params.new_latent_lr;
            output.inverse_temp = params.inverse_temp;
            %output.reward_prior = params.reward_prior;
            if isfield(params, 'decay')
                output.decay = params.decay;
                
            end
            if isfield(params, 'forget_threshold')
                output.froget_threshold = params.forget_threshold;
            end
        
            output.accuracy = accuracy;
            output.action_accuracy = action_accuracy;
            output.LL = L;
            output.free_energy = CPD_fit_output.F; 
        
        writetable(struct2table(output), filename);
        end
    end
end