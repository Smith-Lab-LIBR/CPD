
    cluster_info = readtable('L:/rsmith/lab-members/rhodson/CPD/CPD_results/subjects_by_cluster.csv');
    target_cluster = 1;
    cluster_subjects = cluster_info.SubjectID(cluster_info.Cluster == target_cluster);
    %folder_path = 'L:/rsmith/lab-members/rhodson/CPD/CPD_results/identifiability/CPD_latent_single_inference_expectation_2rl';
    folder_path = 'L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/smaller_comp';
    % Get list of subfolders in the given folder_path
    subfolders = dir(folder_path);
    %subfolders = subfolders([subfolders.isdir] & ~ismember({subfolders.name}, {'.', '..'}));

    % Loop through each subfolder
    %for sf = 1:length(subfolders)
        %subfolder_name = subfolders(sf).name;
        %subfolder_path = fullfile(folder_path, subfolder_name);

        %csv_files = dir(fullfile(subfolder_path, '*.csv'));
        csv_files = dir(fullfile(folder_path, '*.csv'));


        % Initialize
        free_energy_data = [];
        subject_list = [];
        model_names = cell(length(csv_files), 1);

        % Read and align free energies
        for i = 1:length(csv_files)
            file_path = fullfile(folder_path, csv_files(i).name);
            data_table = readtable(file_path);

            % Remove excluded subjects
            data_table(ismember(data_table.subject, {'AA581', 'AJ027' 'AS591'}), :) = [];

            % Uncomment if filtering by cluster
            data_table = data_table(ismember(data_table.subject, cluster_subjects), :);

            % Save subject list on first pass
            if i == 1
                subject_list = data_table.subject;
            else
                assert(isequal(subject_list, data_table.subject), ...
                    'Subject mismatch between model files in folder %s.', folder_path);
            end

            free_energy_data(:, i) = data_table.free_energy;

            % Clean model name
            [~, name, ~] = fileparts(csv_files(i).name);
            model_names{i} = strrep(name, 'individual_CPD_', '');
        end

        % Run BMS
        [alpha, exp_r, xp, pxp, bor] = spm_BMS(free_energy_data);
        post = exp(free_energy_data) ./ sum(exp(free_energy_data), 2);

        % Create results table
        results_table = table(model_names, alpha', exp_r', xp', pxp', ...   
            'VariableNames', {'Model', 'Posterior_Probability', 'Expected_Frequency', 'Exceedance_Probability', 'Protected_Exceedance_Probability'});

        % Save results
        output_file = fullfile(folder_path, sprintf('BMS_results_cluster%d.csv', target_cluster));
            
            
        writetable(results_table, output_file, 'Delimiter', ',');

        %disp(['BMS results for folder ', subfolder_name, ' saved to: ', output_file]);
    %end

