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
%clear
%close all

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
% addpath('/Volumes/labs/rsmith/lab-members/clavalley/MATLAB/spm12/')
% addpath('/Volumes/labs/rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/')

% error_messages = {};

% try
function [] = main_CRP_test(subject_id)

    if ispc
        root = 'C:/';
        %root = 'L:/';
    elseif ismac
        root = '/Volumes/labs/';
    elseif isunix 
        root = '/media/labs/';  
    end

    seed = subject_id(end-2:end);
    seed = str2double(seed);
    rng(seed)

    % for i = 1:numel(all_sub_ids)
    %     subject_id = all_sub_ids{i};
      % try 
    % data_dir = "/Volumes/labs/rsmith/lab-members/nli/CPD_updated/Individual_file_mat";
    % data_dir = "/media/labs/rsmith/lab-members/nli/CPD_updated/Individual_file_mat";
    data_dir = [root 'rsmith/lab-members/nli/CPD_updated/Individual_file_mat'];
    filename = fullfile(data_dir, [subject_id '_trials_CPD_indv.mat']);
    load(filename);

    trial_num = 1;
    while(trial_num <= numel(trials)) %&& all(strcmp(trials{trial_num}.id,'AA181')))
        MDP.trials{trial_num} = trials{trial_num};
        trial_num = trial_num + 1;
    end
    %%%%%%%%%%%%%%
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/']);
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/']); 
    % addpath('/Volumes/labs/rsmith/lab-members/clavalley/MATLAB/spm12/');
    % addpath('/Volumes/labs/rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/'); 
    % %cd("/media/labs/rsmith/lab-members/nli/CPD/matlab_scripts/")
    %%%%% Set Priors %%%%%%%
    reward_lr = 0.1;
    % latent_lr = 0.5;
    % new_latent_lr = 0.1;
    inverse_temp = 2;
    reward_prior = 0;
    alpha = 3   ; 
    
    %% Fit each subject and keep the list of Free energy, 
    % all_sub_ids = readtable('/media/labs/rsmith/lab-members/nli/CPD_updated/T475_list.csv');
    % all_sub_ids = table2cell(all_sub_ids);
    % data_dir = "/Volumes/labs/rsmith/lab-members/nli/CPD_updated/Individual_file_mat";
    F_CRP_model = [];
    LL_CRP_model = [];
    ActionAccu_CRP_model = [];
    Accuracy_CRP_model = [];
    DCM.MDP.reward_lr = reward_lr;
    % DCM.MDP.latent_lr = latent_lr;
    % DCM.MDP.new_latent_lr = new_latent_lr;
    DCM.MDP.inverse_temp = inverse_temp;
    DCM.MDP.reward_prior = reward_prior;
    DCM.MDP.alpha = alpha; 
    DCM.field  = {'reward_lr' 'inverse_temp' 'reward_prior' 'alpha'}; % Parameter field
    DCM.U = MDP.trials;
    DCM.Y = 0;
    CPD_fit_output= CPD_CRP_fit_test(DCM);
    output_params = [1/(1+exp(-CPD_fit_output.Ep.reward_lr)) exp(CPD_fit_output.Ep.inverse_temp) CPD_fit_output.Ep.reward_prior exp(CPD_fit_output.Ep.alpha)];
    
    % we have the best fit model parameters. Simulate the task one more time to
    % get the average action probability and accuracy with these best-fit
    % parameters
    
    params.reward_lr = output_params(1);
    % params.latent_lr = output_params(2);
    % params.new_latent_lr = output_params(3);
    params.inverse_temp = output_params(2);
    params.reward_prior = output_params(3);
    params.alpha = output_params(4);

    
    % rerun model a final time
    L = 0;
    action_probabilities = CRP_RW_Basic_model(params, trials, 0); 
    count = 0;
    average_accuracy = 0;
    average_action_probability = 0;
    accuracy_count = 0;
    % compare action probabilities returned by the model to actual actions
    % taken by participant (as we do in Loss function in CPD_fit
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
    
    %% Store values to the subject
    
    F_CRP_model = [F_CRP_model; cellstr(subject_id), CPD_fit_output.F];
    LL_CRP_model = [LL_CRP_model; cellstr(subject_id), L];
    ActionAccu_CRP_model = [ActionAccu_CRP_model; cellstr(subject_id), action_accuracy];
    Accuracy_CRP_model = [Accuracy_CRP_model; cellstr(subject_id), accuracy];

    % file_name = sprintf('/Volumes/labs/rsmith/lab-members/nli/CPD_updated/Output_Individual_mat_for_all_models/Output_Individual_mat_CRP/%s_individual.mat', subject_id);
    file_name = sprintf('/media/labs/rsmith/lab-members/nli/CPD_updated/Output_Individual_mat_for_all_models/Output_Individual_mat_CRP/%s_individual_CRP.mat', subject_id);
    save(file_name)
    output.subject = subject_id;
    output.reward_lr = output_params(1);
    %output(i).latent_lr = CPD_fit_output.Ep.latent_lr;
    %output(i).new_latent_lr = CPD_fit_output.Ep.new_latent_lr;
    output.inverse_temp = output_params(2);
    output.reward_prior = output_params(3);
    output.alpha = output_params(4);
    output.accuracy = accuracy;
    output.action_accuracy = action_accuracy;
    output.LL = L;
    output.free_energy = CPD_fit_output.F; 

  % catch ME
    % error_messages{end+1} = sprintf('An unexpected error occurred for subject %s: %s\n', subject_id, ME.message);
      
  % end
% end


% filename = sprintf('/Volumes/labs/rsmith/lab-members/nli/CPD_updated/Output_Individual_csv_for_all_models/Output_Individual_csv_CRP/%s_individual.csv', subject_id);
filename = sprintf('/media/labs/rsmith/lab-members/nli/CPD_updated/Output_Individual_csv_for_all_models/Output_Individual_csv_CRP/%s_individual_CRP.csv', subject_id);

writetable(struct2table(output), filename);
end
% catch ME
    % error_messages{end+1} = sprintf('An unexpected error occurred: %s\n', subject_id, ME.message);
% end

