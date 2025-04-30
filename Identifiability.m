function [] = Identifiability(subject_id)
    seed = subject_id(end-2:end);
    DCM.sim = true;
    DCM.use_DDM = true;
    seed = str2double(seed);
    rng(seed);
    
    if ispc
        %root = 'C:/';
        root = 'L:/';
    elseif isunix 
        root = '/media/labs/';  
    end
     addpath(['../..']);
    trial_num = 1;
    trials = load_CPD_data(root,subject_id); 
    while(trial_num <= numel(trials)) %&& all(strcmp(trials{trial_num}.id,'AA181')))
        MDP.trials{trial_num} = trials{trial_num};
        trial_num = trial_num + 1;
    end
    addpath(genpath([root 'rsmith/lab-members/rhodson/CPD/CPD_code']))
    %%%%%%%%%%%%%%
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/']);
    addpath([root 'rsmith/lab-members/clavalley/MATLAB/spm12/toolbox/DEM/']);  
    folder_path = 'L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/DDM';
    
    %outer_fit_list = {@CPD_latent_single_inference_expectation, @CPD_latent_single_inference_max, @CPD_latent_multi_inference_expectation, @CPD_latent_multi_inference_max, @CPD_CRP_multi_inference_expectation, @CPD_CRP_single_inference_expectation, @CPD_RW_Model, @CPD_RW_single};
    %outer_fit_list = {@CPD_latent_single_inference_expectation};
    outer_fit_list = {@CPD_RW_single};
    %outer_fit_list = {@CPD_latent_single_inference_max};
    %inner_fit_list = {'vanilla', 'basic', 'temporal', 'basic_forget', 'temporal_forget'};
    inner_fit_list = {'basic'};
    for i = 1:length(outer_fit_list)
        model_handle = outer_fit_list{i};
        file_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/combined/DDM/%s.csv'], func2str(model_handle));
        model_file = readtable(file_name);
        for j = 1:length(inner_fit_list)
             if exist('DCM', 'var') && isfield(DCM, 'MDP')
                DCM = rmfield(DCM, 'MDP');
             end
         
             % 
             % if isequal(outer_fit_list{i}, @single_inference_expectation)
             %     model_handle = @latent_single_inference_expectation_simulated;
             %     DCM.model = @latent_single_inference_expectation_recov;
             % elseif isequal(outer_fit_list{i}, @single_inference_max)
             %     model_handle = @latent_single_inference_max_simulated;
             %     DCM.model = @latent_single_inference_max_recov;
             % end
             % params.nondecision_time = 0.2;
             params = struct();
             if DCM.use_DDM
                ddm_mappings = {
                    %struct('drift', 'action_prob', 'bias', 'action_prob');
                    struct('drift', '',           'bias', 'action_prob');
                    %struct('drift', 'action_prob', 'bias', '')
                };
                else
                ddm_mappings = {
                    struct('drift', '', 'bias', '');
                };
             end
             if DCM.use_DDM
                    ddm_mapping_string = ['ddm_mapping' 1];
             else
                    ddm_mapping_string = '';
             end
             DCM.drift_mapping = ddm_mappings{1}.drift;
             DCM.bias_mapping = ddm_mappings{1}.bias; 
             idx = strcmp(model_file.subject, subject_id);
             subject_params = model_file(idx,:);
             decay_type = inner_fit_list{j};             
             exclude = {'subject'};  % add other fields to ignore here
            for  z = 1:numel(subject_params.Properties.VariableNames)
                var_name = subject_params.Properties.VariableNames{z};
                if ~ismember(var_name, exclude)
                    params.(var_name) = subject_params.(var_name);
                end
            end

             %% get behavior from model
            model_output = model_handle(params, trials, decay_type, DCM);

            %% Now fit all models to this behavior
            rl_identifiability(subject_id, model_output.simmed_choices, func2str(model_handle))
            %CRP_identifiability(subject_id, model_output.simmed_choices, func2str(model_handle))
            %latent_learning_identifiability(subject_id, model_output.simmed_choices, func2str(model_handle))
            %folder_name = sprintf([root 'rsmith/lab-members/rhodson/CPD/CPD_results/identifiability/%s/'],func2str(model_handle));
           % generate_pxp(folder_name, func2str(model_handle))
        end
    end
end