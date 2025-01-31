function v = merge_vectors(vec1, vec2)

    maxLength = max(length(vec1), length(vec2));
    % Extend both vectors to the maximum length
    vec1_padded = [vec1, zeros(1, maxLength - length(vec1))];
    vec2_padded = [vec2, zeros(1, maxLength - length(vec2))];
    
    % Sum the vectors
    v = vec1_padded + vec2_padded;
end
