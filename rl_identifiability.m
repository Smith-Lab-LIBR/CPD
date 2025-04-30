
function [] = rl_identifiability(subject_id, trials, generating_model)
     seed = subject_id(end-2:end);
    seed = str2double(seed);
    rng(seed);
    DCM.use_DDM = true;

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
    reward_lr = 0.5;
    % latent_lr = 0.5;
    % new_latent_lr = 0.1;
    inverse_temp = 1;
    reward_prior = 0;
    decay = 0.8;
   outer_fit_list = {@CPD_RW_single};
    inner_fit_list = {'basic'};
    F_CRP_model = [];
    LL_CRP_model = [];
    ActionAccu_CRP_model = [];
    Accuracy_CRP_model = [];
    for i = 1:length(outer_fit_list)
        model_handle = outer_fit_list{i};
        for j = 1:length(inner_fit_list)
            if DCM.use_DDM
                ddm_mappings = {
                    %struct('drift', 'action_prob', 'bias', 'action_prob');
                    %struct('drift', '',           'bias', 'action_prob');
                    struct('drift', 'action_prob', 'bias', '')
                };
            else
                ddm_mappings = {
                    struct('drift', '', 'bias', '');
                };
            end
            for mapping_index=1:length(ddm_mappings)
                 if exist('DCM', 'var') && isfield(DCM, 'MDP')
                    DCM = rmfield(DCM, 'MDP');
                 end

                if strcmp(inner_fit_list{j}, 'vanilla')
                    decay_type = "";
                else
                    decay_type = inner_fit_list{j};
                end
    
                DCM.MDP.reward_lr = reward_lr;
                %DCM.MDP.existing_latent_lr = existing_latent_lr;
                DCM.MDP.inverse_temp = inverse_temp;
                DCM.MDP.reward_prior = reward_prior;
                DCM.model = model_handle;
                if DCM.use_DDM
                    ddm_mapping_string = ['ddm_mapping' num2str(mapping_index)];
                else
                    ddm_mapping_string = '';
                end
                folder_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/identifiability/DDM/%s/rl/'], generating_model);
                if ~exist(folder_name, 'dir')
                    mkdir(folder_name);
                end
                if strcmp(inner_fit_list{j}, 'vanilla')
                     DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior'}; % Parameter field
                     
                     filename = sprintf([folder_name '%s_individual_%s.csv'], subject_id, func2str(DCM.model));
                else
                    DCM.MDP.decay = decay;
                    DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior' 'decay' }; % Parameter field
                  
                    filename = sprintf([folder_name '%s_individual_%s_%s.csv'], subject_id, func2str(DCM.model), decay_type);
                end
                %%% set up DDM 
                if DCM.use_DDM
                    DCM.max_rt = 2;
                    DCM.min_rt = .3;
                    
                    DCM.drift_mapping = ddm_mappings{mapping_index}.drift; % specify that action_prob maps to drift or leave blank so that drift rate will be fit as free parameter
                    DCM.bias_mapping = ddm_mappings{mapping_index}.bias; % specify that action_prob maps to starting bias or leave blank so that starting bias will be fit as free parameter
    
                    if strcmp(DCM.drift_mapping,'action_prob')
                        DCM.MDP.drift_baseline = .085; % parameter for baseline drift rate
                        DCM.field{end+1} = 'drift_baseline'; 
                        DCM.MDP.drift_mod = .5; % parameter that maps action probability onto drift rate
                        DCM.field{end+1} = 'drift_mod';
                    else
                        DCM.MDP.drift = 0; 
                        DCM.field{end+1} = 'drift';
                    end
                    
                    if strcmp(DCM.bias_mapping,'action_prob')
                        DCM.MDP.bias_mod = .5; % parameter that maps action probability onto starting bias
                        DCM.field{end+1} = 'bias_mod';
                    else
                        DCM.MDP.starting_bias = .5;
                        DCM.field{end+1} = 'starting_bias';
                    end
                    DCM.MDP.nondecision_time = .2;
                    DCM.field{end+1} = 'nondecision_time';
                    DCM.MDP.decision_thresh = 2;
                    DCM.field{end+1} = 'decision_thresh';
    
                end
                DCM.U = MDP.trials;
                DCM.Y = 0;
                DCM.sim = false;
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
                if isfield(CPD_fit_output.Ep, 'drift_baseline')
                    params.drift_baseline = CPD_fit_output.Ep.drift_baseline;
                end
                if isfield(CPD_fit_output.Ep, 'drift')
                    params.drift = CPD_fit_output.Ep.drift;
                end
    
                if isfield(CPD_fit_output.Ep, 'starting_bias')
                    params.starting_bias = 1/(1+exp(-CPD_fit_output.Ep.starting_bias));
                end
                if isfield(CPD_fit_output.Ep, 'drift_mod')
                    params.drift_mod = 1/(1+exp(-CPD_fit_output.Ep.drift_mod));
                end
                if isfield(CPD_fit_output.Ep, 'bias_mod')
                    params.bias_mod = 1/(1+exp(-CPD_fit_output.Ep.bias_mod));
                end
                if isfield(CPD_fit_output.Ep, 'decision_thresh')
                    params.decision_thresh = exp(CPD_fit_output.Ep.decision_thresh);
                end
                if isfield(CPD_fit_output.Ep, 'nondecision_time')
                    params.nondecision_time = 0.1 + (0.3 - 0.1) ./ (1 + exp(-CPD_fit_output.Ep.nondecision_time)); 
                end
    
            
                % rerun model a final time
                L = 0;
                model_output = DCM.model(params, trials, decay_type, DCM); 
                action_probabilities = model_output.patch_action_probs;
                count = 0;
                average_accuracy = 0;
                average_action_probability = 0;
                accuracy_count = 0;
                % compare action probabilities returned by the model to actual actions
                % taken by participant (as we do in Loss function in CPD_fit
                rng(1)
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
                if DCM.use_DDM
                    rt_pdf = model_output.dot_motion_rt_pdf;
                    all_values = rt_pdf(~isnan(rt_pdf(:)));
                    L = L + sum(log(all_values + eps));
                end
                %These are the final values. 
                action_accuracy = average_action_probability/count;
                accuracy = average_accuracy/accuracy_count;
                
                fprintf('Final LL: %f \n',L)
                fprintf('Final Average choice probability: %f \n',action_accuracy)
                fprintf('Final Average Accuracy: %f \n',accuracy)          
              
                %save(file_name)
                output.subject = subject_id;
                flds = fieldnames(params);
                for z = 1:numel(flds)
                    output.(flds{z}) = params.(flds{z});
                end
                % output.subject = subject_id;
                % output.reward_lr = params.reward_lr;
                % output.inverse_temp = params.inverse_temp;
                % output.reward_prior = params.reward_prior;
                % if isfield(params, 'decay')
                %     output.decay = params.decay;
                % end            
            
                output.patch_choice_avg_action_prob = action_accuracy;
                output.patch_choice_model_acc = accuracy;
                output.dot_motion_avg_action_prob = mean(model_output.dot_motion_action_prob(~isnan(model_output.dot_motion_action_prob)));
                output.dot_motion_model_acc = mean(model_output.dot_motion_model_acc(~isnan(model_output.dot_motion_model_acc)));
                output.LL = L;
                output.free_energy = CPD_fit_output.F; 
            
                writetable(struct2table(output), filename);
            end
        end
    end
end