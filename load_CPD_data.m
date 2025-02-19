function data = load_CPD_data(root,subject_id)

    data_dir = [root '/NPC/Analysis/T1000/data-organized/' subject_id '/T0/behavioral_session/']; % always in T0?

    has_practice_effects = false;
    % Manipulate Data
    directory = dir(data_dir);
    % sort by date
    dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    % Sort the dates and get the sorted indices
    [~, sortedIndices] = sort(dates);
    % Use the sorted indices to sort the structure array
    sortedDirectory = directory(sortedIndices);
    index_array = find(arrayfun(@(n) contains(sortedDirectory(n).name, 'CPD-R1-_BEH'),1:numel(sortedDirectory)));
    if length(index_array) > 1
        disp("WARNING, MULTIPLE BEHAVIORAL FILES FOUND FOR THIS ID. USING THE FIRST FULL ONE")
    end

    for k = 1:length(index_array)
        file_index = index_array(k);
        file = [data_dir sortedDirectory(file_index).name];

        subdat = readtable(file);
        % Practice is usually 60 trials but can be more/less. Games will always be 290 trials            
        % event code 13 signals final score shown. complete file!!!
        if any(subdat.event_code == 13)
            break;
        else
            % if the participant doesn't have any more files to look
            % through, throw an error because did not finish
            if k == length(index_array)
                error("Participant does not have a complete behavioral file");
            else
                % if they made it passed trial 60, indicate that they have
                % practice effects
                if ~strcmp(class(subdat.trial_number),'double')
                    continue;
                else
                    max(subdat.trial_number) >= 60
                    has_practice_effects = true;
                end
                continue;
            end
        end
    end
    last_practice_trial = max(subdat.trial_number) - 290;
    first_game_trial = min(find(subdat.trial_number == last_practice_trial+1));
    clean_subdat = subdat(first_game_trial:end, :);
    
    
    clean_subdat_filtered = clean_subdat(clean_subdat.event_code==7 | clean_subdat.event_code==8 | clean_subdat.event_code==9,:);
    DCM.behavioral_file = clean_subdat_filtered;
    
    % convert columns to double where applicable
    colsToConvert = {'response_time', 'result', 'response'};
    for i = 1:length(colsToConvert)
        colName = colsToConvert{i};
        if iscellstr(clean_subdat_filtered.(colName)) || isstring(clean_subdat_filtered.(colName))
            clean_subdat_filtered.(colName) = str2double(clean_subdat_filtered.(colName));
        end
    end
    
    % take the last 290 trials
    % event code 7 is game onset, event code 8 means they open a patch, event code 9 means they
    % accept dot motion.
    % note that the result column indicates the correct patch for the first
    % row. For the last row, it indicates if participant chose the correct
    % patch (1) or incorrect patch (0)
    data = cell(1,290);
    for (trial_number=1:290)
        game = clean_subdat_filtered(clean_subdat_filtered.trial_number == trial_number+last_practice_trial,:);
        game.accept_reject_rt = nan(height(game),1);
        for (row=2:height(game)-1)
            game.accept_reject_rt(row) = game.response_time(row+1) - game.response_time(row);
        end
        game = game(1:end-1,:);
        % note that participants must accept a dot motion for trial to
        % continue
        game.accepted_dot_motion = zeros(height(game), 1); % Set all values to 0
        game.accepted_dot_motion(end) = 1; % set last value to 1
        
        data(trial_number) = {game};
    end


    % re-order table so that event_code is first, then response, then
    % result
    for trial_number = 1:290
        if ~isempty(data{trial_number})
            desired_order = {'event_code', 'response', 'result'};
            existing_columns = data{trial_number}.Properties.VariableNames;
            remaining_columns = setdiff(existing_columns, desired_order, 'stable');
            new_order = [desired_order, remaining_columns];
            data{trial_number} = data{trial_number}(:, new_order);
        end
    end



end