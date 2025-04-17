% %% Read all participants 
% clear
% close all
% 
% multi_action = 1;
% 
% SUFFIX = "_trials_CPD_indv";
% all_sub_ids = readtable('/Users/nli/Desktop/CPD/Data/T475_list.csv');
% all_sub_ids = table2cell(all_sub_ids);
% out_directory = "/Users/nli/Desktop/CPD/Data/Individual_file_mat/";
% base_path = "/Users/nli/Desktop/CPD/Data/processed_real_trials/"; 
% file_paths = cell(1, numel(all_sub_ids)); 
% 
% for i = 1:numel(all_sub_ids)
%     file_paths{i} = fullfile(base_path,[all_sub_ids{i} '-T0-_CPD-R1-_BEH.csv']);
% end
% 
% 
% for i = 1:numel(file_paths)
%    try
%     CPD = readtable(file_paths{i});
%     [~, filename, ~] = fileparts(file_paths{i});
%     parts = split(filename, '-');
%     subject_id = parts{1};
%     CPD = CPD(:, {'event_code', 'response', 'result'});
%     % Read the CSV file into a matrix
%     third_column = CPD{:, 1}; % event_code 
%     % Convert the table to a matrix
% 
%     % Extract rows with the value 8 in the third column
%     CPD = CPD(third_column == 8 | third_column == 9, :); %7-target (0,1,2-> left, bottom, or right ) 
%                                                          %8-search sequence
%                                                          %9-accuracy
%         %1-trial number
%         %2-trial type (1-6, 1:3:9, 7, 1:1:1);
%         %3-event code
%            %7-target (0,1,2-> left, bottom, or right )
%            %8-search sequence
%            %9-accuracy
%     CPD_data = CPD(:, {'event_code','response', 'result'});
% 
%         varNames = {'event_code','response', 'result'};
%         varTypes = {'double','double', 'double'};
%         num_rows = height(CPD_data);
%         % CPD_data.response = str2double(CPD_data.response);
%         trials = {};
%         trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
%         if multi_action == 1    
%             for i = 1:num_rows
%                 row = CPD_data(i,:);
% 
%                 if row.event_code ~= 9
%                     trial = [trial; row];
%                 else
%                     trials{end+1} = trial;
%                     trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
%                 end
%             end
% 
%         else
%             for i = 1:num_rows
%                 row = CPD_data(i,:);
%                 trial = [trial; row];
%                 trials{end+1} = trial;
%                 trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
%             end
%         end
% 
%         output_filename = out_directory + subject_id + SUFFIX;
%         save(output_filename, 'trials'); 
% 
% 
%   catch ME
%     disp(['An error occurred: ', subject_id, ME.message]);
%   end
% 
% end

% %% Single Subject
% 
%%%%% Read in Data %%%%%%


% rng('shuffle')
% multi_action = 1;
% if ispc
%     root = 'C:/';
%     %root = 'L:/';
% elseif isunix 
%     root = '/media/labs/';  
% end

% 
% %CPD = readtable('L:/rsmith/lab-members/rhodson/CPD/AD421-T0-_CPD-R1-_BEH');
% CPD = readtable('/Users/nli/Desktop/CPD/AC609-T0-_CPD-R1-_BEH.csv');
% CPD = CPD(:, {'event_code', 'response', 'result'});
% % Read the CSV file into a matrix
% third_column = CPD{:, 1};
% % Convert the table to a matrix
% 
% % Extract rows with the value 8 in the third column
% CPD = CPD(third_column == 8 | third_column == 9, :);
% CPD_data = CPD(:, {'event_code','response', 'result'});
% 
%     varNames = {'event_code','response', 'result'};
%     varTypes = {'double','double', 'double'};
%     num_rows = height(CPD_data);
%     CPD_data.response = str2double(CPD_data.response);
%     trials = {};
%     trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
%     if multi_action == 1    
%         for i = 1:num_rows
%             row = CPD_data(i,:);
% 
%             if row.event_code ~= 9
%                 trial = [trial; row];
%             else
%                 trials{end+1} = trial;
%                 trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
%             end
%         end
% 
%     else
%         for i = 1:num_rows
%             row = CPD_data(i,:);
%             trial = [trial; row];
%             trials{end+1} = trial;
%             trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
%         end
%     end
%     save('CPD_trials.mat', 'trials'); 
% 
% 
% 
% % if ~exist('trials', 'var')
% %     trials = load('CPD_trials.mat', 'trials').trials;
% % else
% %     if ispc
% %         CPD = readtable('L:/rsmith/lab-members/rhodson/CPD/df_reaction_time.csv');
% %     else
% %         CPD = readtable([root '/rsmith/lab-members/rhodson/CPD/df_reaction_time.csv'], 'FileType', 'text');
% %     end
% % 
% % 
% %     CPD_data = CPD(:,{'response', 'result', 'id', 'LC_Category'});
% %     CPD_data = CPD(:,{'response', 'result', 'id', 'LC_Category'});  
% %     varNames = {'response', 'result', 'id', 'LC_Category'};
% %     varTypes = {'double', 'double', 'cell', 'cell'};
% %     num_rows = 350;%height(CPD_data);
% %     trials = {};
% %     trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
% %     if multi_action == 1    
% %         for i = 1:num_rows
% %             row = CPD_data(i,:);
% %             trial = [trial; row];
% %             if height(trial) == 3 || row.result == 1
% %                 trials{end+1} = trial;
% %                 trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
% %             end
% %         end
% % 
% %     else
% %         for i = 1:num_rows
% %             row = CPD_data(i,:);
% %             trial = [trial; row];
% %             trials{end+1} = trial;
% %             trial = table('Size', [0, numel(varNames)],'VariableNames', varNames, 'VariableTypes', varTypes);
% %         end
% %     end
% %       save('CPD_trials.mat', 'trials'); 
% %   end
% 
% %addpath([root '/rsmith/all-studies/util/spm12/']);
% addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);


function [] = main(subject_id) % main('AA181')
    DCM.use_DDM = true;
    seed = subject_id(end-2:end);
    seed = str2double(seed);
    rng(seed);
    
    if ispc
        %root = 'C:/';
        root = 'L:/';
    elseif isunix 
        root = '/media/labs/';  
    end

    % for i = 1:numel(all_sub_ids)
    %     subject_id = all_sub_ids{i};
      % try 
    % data_dir = "/Volumes/labs/rsmith/lab-members/nli/CPD_updated/Individual_file_mat";
    %  data_dir = [root 'rsmith/lab-members/nli/CPD_updated/Individual_file_mat'];
    %  filename = fullfile(data_dir, [subject_id '_trials_CPD_indv.mat']);
    % load(filename);
    trials = load_CPD_data(root,subject_id);

    trial_num = 1;
    while(trial_num <= numel(trials)) %&& all(strcmp(trials{trial_num}.id,'AA181')))
        MDP.trials{trial_num} = trials{trial_num};
        trial_num = trial_num + 1;
    end
    %%%%%%%%%%%%%%
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/']);
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/']);  

    %cd("/media/labs/rsmith/lab-members/nli/CPD/matlab_scripts/")
    %%%%% Set Priors %%%%%%%
    reward_lr = 0.1;
    % latent_lr = 0.5;
    % new_latent_lr = 0.1;
    inverse_temp = 1;
    reward_prior = 0;
    decay = 0.8;
   outer_fit_list = {@CPD_RW_single};
    inner_fit_list = {'vanilla', 'basic'};
    F_CRP_model = [];
    LL_CRP_model = [];
    ActionAccu_CRP_model = [];
    Accuracy_CRP_model = [];
    for i = 1:length(outer_fit_list)
        model_handle = outer_fit_list{i};
        for j = 1:length(inner_fit_list)
            if DCM.use_DDM
                ddm_mappings = {
                    struct('drift', 'action_prob', 'bias', 'action_prob');
                    struct('drift', '',           'bias', 'action_prob');
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
                if DCM.use_DDM
                    ddm_mapping_string = ['ddm_mapping' num2str(mapping_index)];
                else
                    ddm_mapping_string = '';
                end
                if j == 1
                     DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior'}; % Parameter field
                     file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/rl/ind_mat/%s_individual_%s_%s.mat'], subject_id, func2str(DCM.model),ddm_mapping_string);
                     filename = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/rl/ind_csv/%s_individual_%s_%s.csv'], subject_id, func2str(DCM.model),ddm_mapping_string);
                else
                    DCM.MDP.decay = decay;
                    DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior' 'decay' }; % Parameter field
                    file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/rl/ind_mat/%s_individual_%s_%s_%s.mat'], subject_id, func2str(DCM.model), decay_type,ddm_mapping_string);
                    filename = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/rl/ind_csv/%s_individual_%s_%s_%s.csv'], subject_id, func2str(DCM.model), decay_type,ddm_mapping_string);
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
              
                save(file_name)
                output.subject = subject_id;
                output.reward_lr = params.reward_lr;
                output.inverse_temp = params.inverse_temp;
                output.reward_prior = params.reward_prior;
                if isfield(params, 'decay')
                    output.decay = params.decay;
                end            
            
                output.patch_choice_avg_action_prob = accuracy;
                output.patch_choice_model_acc = action_accuracy;
                output.dot_motion_avg_action_prob = mean(model_output.dot_motion_action_prob(~isnan(model_output.dot_motion_action_prob)));
                output.dot_motion_model_acc = mean(model_output.dot_motion_model_acc(~isnan(model_output.dot_motion_model_acc)));
                output.LL = L;
                output.free_energy = CPD_fit_output.F; 
            
                writetable(struct2table(output), filename);
            end
        end
    end
end