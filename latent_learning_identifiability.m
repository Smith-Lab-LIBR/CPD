function [] = latent_learning_identifiability(subject_id, trials, generating_model)
    DCM.use_DDM = true;

    dbstop if error; 
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
    addpath(['../..']); % add parent of parent directory to path
    %%%%%%%%%%%%%%
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/']);
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/']); 
    addpath(genpath([root 'rsmith/lab-members/rhodson/CPD/CPD_code/Latent_Learning/Latent_learning_Models']))
    % addpath('/Volumes/labs/rsmith/lab-members/clavalley/MATLAB/spm12/');
    % addpath('/Volumes/labs/rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/'); 
    % %cd("/media/labs/rsmith/lab-members/nli/CPD/matlab_scripts/")
    %%%%% Set Priors %%%%%%%
   
    %alpha = 1; 
    %% Fit each subject and keep the list of Free energy, 
    % all_sub_ids = readtable('/media/labs/rsmith/lab-members/nli/CPD_updated/T475_list.csv');
    % all_sub_ids = table2cell(all_sub_ids);
    % data_dir = "/Volumes/labs/rsmith/lab-members/nli/CPD_updated/Individual_file_mat";
    outer_fit_list = {@CPD_latent_single_inference_expectation};
    % outer_fit_list = {@CPD_latent_multi_inference_m};

    %outer_fit_list = {@CPD_latent_single_inference_expectation, @CPD_latent_single_inference_max};
    %outer_fit_list = {@CPD_latent_single_inference_expectation, @CPD_latent_single_inference_max, @CPD_latent_multi_inference_max};
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
            

            DCM.MDP.reward_lr = reward_lr;
            DCM.MDP.latent_lr = latent_lr;
            DCM.MDP.new_latent_lr = new_latent_lr;
            %DCM.MDP.existing_latent_lr = existing_latent_lr;
            DCM.MDP.inverse_temp = inverse_temp;
            DCM.MDP.reward_prior = reward_prior;
            DCM.model = model_handle;
         
            %%% set up DDM 
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
                reward_lr = 0.5;
                latent_lr = 0.5;
                new_latent_lr = 0.5;
                inverse_temp = 1;
                reward_prior = 0;
                new_reward_lr = 0.5;
                decay = 0.9;
                forget_threshold = 0.05;
                 if strcmp(inner_fit_list{j}, 'vanilla')
                     decay_type = "";
                 elseif strcmp(inner_fit_list{j}, 'basic_forget')
                    decay_type = "basic";
                 elseif strcmp(inner_fit_list{j}, 'temporal_forget')
                    decay_type = "temporal";
                 else
                     decay_type = inner_fit_list{j};
                 end
                DCM.MDP.new_reward_lr = new_reward_lr;
                DCM.MDP.reward_lr = reward_lr;
                DCM.MDP.latent_lr = latent_lr;
                DCM.MDP.new_latent_lr = new_latent_lr;
                %DCM.MDP.existing_latent_lr = existing_latent_lr;
                DCM.MDP.inverse_temp = inverse_temp;
                DCM.MDP.reward_prior = reward_prior;
                DCM.model = model_handle;
                if DCM.use_DDM
                    ddm_mapping_string = ['ddm_mapping' num2str(mapping_index)];
                else
                    ddm_mapping_string = '';
                end
                folder_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/identifiability/DDM/%s/latent_learning/'], generating_model);
                if ~exist(folder_name, 'dir')
                    mkdir(folder_name);
                end
                if strcmp(inner_fit_list{j}, 'vanilla')
                     %DCM.field  = {'reward_lr' 'inverse_temp' 'latent_lr'}; % Parameter field
                     DCM.field  = {'reward_lr' 'reward_prior' 'inverse_temp' 'latent_lr'};
                     filename = sprintf('%s_%s.csv', subject_id, func2str(DCM.model));
                else
                    DCM.MDP.decay = decay;
                    DCM.MDP.forget_threshold = forget_threshold; 
                    DCM.field  = {'reward_lr' 'new_reward_lr' 'inverse_temp' 'latent_lr' 'new_latent_lr', 'decay', 'forget_threshold'}; % Parameter field
                    file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/ind_mat/%s_individual_%s_%s_forget.mat'], subject_id, func2str(DCM.model), decay_type);
                    filename = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/threshold/%s_individual_%s_%s_forget.csv'], subject_id, func2str(DCM.model), decay_type);
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
                CPD_fit_output= CPD_latent_fit(DCM);
                
                % we have the best fit model parameters. Simulate the task one more time to
                % get the average action probability and accuracy with these best-fit
                % parameters
                params = struct();
                if isfield(CPD_fit_output.Ep, 'reward_lr')
                    params.reward_lr = 1/(1+exp(-CPD_fit_output.Ep.reward_lr));
                end
                if isfield(CPD_fit_output.Ep, 'new_reward_lr')
                    params.new_reward_lr= 1/(1+exp(-CPD_fit_output.Ep.new_reward_lr));
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
                subj_action_probs = [];
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
    
                
                %These are the final values. 
                action_accuracy = average_action_probability/count;
                accuracy = average_accuracy/accuracy_count;
                if DCM.use_DDM
                    rt_pdf = model_output.dot_motion_rt_pdf;
                    all_values = rt_pdf(~isnan(rt_pdf(:)));
                    L = L + sum(log(all_values + eps));
                end
                fprintf('Final LL: %f \n',L)
                fprintf('Final Average choice probability: %f \n',action_accuracy)
                fprintf('Final Average Accuracy: %f \n',accuracy)          
              
                %save(file_name)
                % output = struct();
                % output.subject = subject_id;
                % output.reward_lr = params.reward_lr;
                % output.latent_lr = params.latent_lr;
                % %output.new_latent_lr = params.new_latent_lr;
                % output.inverse_temp = params.inverse_temp;
                % %output.reward_prior = params.reward_prior;
                % if isfield(params, 'decay')
                %     output.decay = params.decay;
                % 
                % end
                % if isfield(params, 'forget_threshold')
                %     output.froget_threshold = params.forget_threshold;
                % end
                % 
                %save(file_name)
                output = struct();
                latent_state_rewards = model_output.reward_probabilities;
                filename_latent_states = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/latent_states/%s_%s_%s_latent_states.csv'], subject_id, func2str(DCM.model), ddm_mapping_string);
                output.subject = subject_id;
    
    
                output.subject = subject_id;
                flds = fieldnames(params);
                for z = 1:numel(flds)
                    output.(flds{z}) = params.(flds{z});
                end


                %output.latent_state_rewards = latent_state_rewards;
                 output.num_states = size(latent_state_rewards,1);
                % output.reward_lr = params.reward_lr;
                % output.latent_lr = params.latent_lr;
                % %output.new_latent_lr = params.new_latent_lr;
                % output.inverse_temp = params.inverse_temp;
                % output.reward_prior = params.reward_prior;
                %     if isfield(params, 'decay')
                %         output.decay = params.decay;
                % 
                %     end
                %     if isfield(params, 'forget_threshold')
                %         output.froget_threshold = params.forget_threshold;
                %     end
          
                % Remove the matrix field from the struct before turning into a table
                %metadata = rmfield(output, 'latent_state_rewards');
    
                % Convert to table (safe now that it's all scalar-compatible)
                %metadata_table = struct2table(metadata, 'AsArray', true);
    
                % Save metadata
                %writetable(metadata_table, 'metadata.csv');
              
                % Save matrix separately
                writematrix(latent_state_rewards, filename_latent_states);
            
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