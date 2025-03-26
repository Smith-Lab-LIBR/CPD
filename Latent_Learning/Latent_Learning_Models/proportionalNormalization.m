function normArr = proportionalNormalization(arr)
    minVal = min(arr);
    adjusted = arr - minVal; % Shift to make all values positive
    normArr = (adjusted + exp(-16)) ./ sum(adjusted +exp(-16),2); % Scale to sum to 1
end