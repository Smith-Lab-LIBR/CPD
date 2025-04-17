
function [] = rl_identifiability(subject_id, trials, generating_model)
     seed = subject_id(end-2:end);
    seed = str2double(seed);
    rng(seed);
    
    if ispc
        %root = 'C:/';
        root = 'L:/';
    elseif isunix 
        root = '/media/labs/';  
    end

    MDP.trials = trials;
    %%%%%%%%%%%%%%
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/']);
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/']);  

    %%%%% Set Priors %%%%%%%
    reward_lr = 0.1;
    % latent_lr = 0.5;
    % new_latent_lr = 0.1;
    inverse_temp = 1;
    reward_prior = 0;
    decay = 0.8;
   outer_fit_list = {@CPD_RW_single, @CPD_RW_Model};
    inner_fit_list = {'vanilla'};
    F_CRP_model = [];
    LL_CRP_model = [];
    ActionAccu_CRP_model = [];
    Accuracy_CRP_model = [];
    for i = 1:length(outer_fit_list)
        model_handle = outer_fit_list{i};
        for j = 1:length(inner_fit_list)
            if j == 1
                decay_type = "";
            else
                decay_type = inner_fit_list{j};
            end

            DCM.MDP.reward_lr = reward_lr;
            %DCM.MDP.existing_latent_lr = existing_latent_lr;
            DCM.MDP.inverse_temp = inverse_temp;
            DCM.MDP.reward_prior = reward_prior;
            DCM.model = model_handle;
            folder_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/identifiability/%s/rl/'], generating_model);
            if ~exist(folder_name, 'dir')
                mkdir(folder_name);
            end
            if j == 1
                 DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior'}; % Parameter field
                 
                 filename = sprintf('%s_individual_%s.csv', subject_id, func2str(DCM.model));
            else
                DCM.MDP.decay = decay;
                DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior' 'decay' }; % Parameter field
              
                filename = sprintf('%s_individual_%s_%s.csv', subject_id, func2str(DCM.model), decay_type);
            end

            DCM.U = MDP.trials;
            DCM.Y = 0;
            DCM.decay_type = decay_type;
            DCM.sim = false;
            CPD_fit_output= CPD_RL_fit(DCM);
            
            % we have the best fit model parameters. Simulate the task one more time to
            % get the average action probability and accuracy with these best-fit
            % parameters
            if isfield(CPD_fit_output.Ep, 'reward_lr')
                params.reward_lr = 1/(1+exp(-CPD_fit_output.Ep.reward_lr));
            end
            if isfield(CPD_fit_output.Ep, 'inverse_temp')
                params.inverse_temp = exp(CPD_fit_output.Ep.inverse_temp);
            end
            if isfield(CPD_fit_output.Ep, 'reward_prior')
                params.reward_prior = CPD_fit_output.Ep.reward_prior;
            end
            if isfield(CPD_fit_output.Ep, 'decay')
                params.decay = 1/(1+exp(-CPD_fit_output.Ep.decay)); 
            end

        
            % rerun model a final time
            L = 0;
            model_output = DCM.model(params, trials, decay_type, DCM); 
            action_probabilities = model_output.action_probs;
            count = 0;
            average_accuracy = 0;
            average_action_probability = 0;
            accuracy_count = 0;
            % compare action probabilities returned by the model to actual actions
            % taken by participant (as we do in Loss function in CPD_fit
            %rng(1)
            for t = 1:length(trials)
                trial = trials{t};
                responses = trial.response;
                first_choice = responses(2);
                if height(trial) > 2       
                    second_choice = responses(3);
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
          
           
            output.subject = subject_id;
            output.reward_lr = params.reward_lr;
            output.inverse_temp = params.inverse_temp;
            output.reward_prior = params.reward_prior;
            if isfield(params, 'decay')
                output.decay = params.decay;
            end            
            output.accuracy = accuracy;
            output.action_accuracy = action_accuracy;
            output.LL = L;
            output.free_energy = CPD_fit_output.F; 
        
        writetable(struct2table(output), [folder_name filename]);
        end
    end
end