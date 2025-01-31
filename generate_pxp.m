% Define the folder containing CSV files
folder_path = 'L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined'; % Change to your folder path
csv_files = dir(fullfile(folder_path, '*.csv')); % Get all CSV files

% Check if files are found
if isempty(csv_files)
    error('No CSV files found in the specified folder.');
end

% Initialize an array to store free energy values
free_energy_data = [];

% Extract model names (file names without .csv extension)
model_names = cell(length(csv_files), 1);

% Process each CSV file (each representing a model)
for i = 1:length(csv_files)
    % Get full file path
    file_path = fullfile(folder_path, csv_files(i).name);
    
    % Read the CSV file into a table
    data_table = readtable(file_path);
    
    % Check if 'free_energy' column exists
    if ~ismember('free_energy', data_table.Properties.VariableNames)
        error(['Column "free_energy" not found in file: ', csv_files(i).name]);
    end
    
    % Extract free energy column
    free_energy = data_table.free_energy; 
    
    % Store free energy data
    free_energy_data(:,i) = free_energy; % Each column is a model

    % Store the model name (remove '.csv' extension and 'individual_')
    [~, model_names{i}, ~] = fileparts(csv_files(i).name);
    model_names{i} = strrep(model_names{i}, 'individual_CPD_', ''); % Remove 'individual_'
end

% Convert free energy to log model evidence
log_evidence = -free_energy_data; % Since Free Energy = -Log Evidence

% Perform Bayesian Model Selection (BMS)
[alpha, exp_r, xp, pxp] = spm_BMS(free_energy_data);

% Create a table for exporting (transpose vectors to match model names)
results_table = table(model_names, alpha', exp_r', xp', pxp', ...
    'VariableNames', {'Model', 'Posterior_Probability', 'Expected_Frequency', 'Exceedance_Probability', 'Protected_Exceedance_Probability'});

% Define CSV output file path
output_file = fullfile('L:/rsmith/lab-members/rhodson/CPD/CPD_results', 'BMS_results.csv');

% Save table as CSV
writetable(results_table, output_file, 'Delimiter', ',');

% Display message
disp(['BMS results saved to: ', output_file]);
