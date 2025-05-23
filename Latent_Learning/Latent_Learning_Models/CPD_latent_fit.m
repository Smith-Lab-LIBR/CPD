% Model fitting script for CPD task
function DCM = CPD_latent_fit(DCM)

% MDP inversion using Variational Bayes
% FORMAT [DCM] = spm_dcm_mdp(DCM)

% If simulating - comment out section on line 196
% If not simulating - specify subject data file in this section 

%
% Expects:
%--------------------------------------------------------------------------
% DCM.MDP   % MDP structure specifying a generative model
% DCM.field % parameter (field) names to optimise
% DCM.U     % cell array of outcomes (stimuli)
% DCM.Y     % cell array of responses (action)
%
% Returns:
%--------------------------------------------------------------------------
% DCM.M     % generative model (DCM)
% DCM.Ep    % Conditional means (structure)
% DCM.Cp    % Conditional covariances
% DCM.F     % (negative) Free-energy bound on log evidence
% 
% This routine inverts (cell arrays of) trials specified in terms of the
% stimuli or outcomes and subsequent choices or responses. It first
% computes the prior expectations (and covariances) of the free parameters
% specified by DCM.field. These parameters are log scaling parameters that
% are applied to the fields of DCM.MDP. 
%
% If there is no learning implicit in multi-trial games, only unique trials
% (as specified by the stimuli), are used to generate (subjective)
% posteriors over choice or action. Otherwise, all trials are used in the
% order specified. The ensuing posterior probabilities over choices are
% used with the specified choices or actions to evaluate their log
% probability. This is used to optimise the MDP (hyper) parameters in
% DCM.field using variational Laplace (with numerical evaluation of the
% curvature).
%
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_mdp.m 7120 2017-06-20 11:30:30Z spm $

% OPTIONS
%--------------------------------------------------------------------------
ALL = false;

% prior expectations and covariance
%--------------------------------------------------------------------------
% parameter list:
% reward_lr
% latent_lr
% new_latent_lr
% inverse_temp
prior_variance = 1;

for i = 1:length(DCM.field)
    field = DCM.field{i};
    try
        param = DCM.MDP.(field);
        param = double(~~param);
    catch
        param = 1;
    end
    if ALL
        pE.(field) = zeros(size(param));
        pC{i,i}    = diag(param);
    else
        if strcmp(field,'reward_lr')
            pE.(field) = log(DCM.MDP.reward_lr/(1-DCM.MDP.reward_lr));           
            pC{i,i}    = 2;
         elseif strcmp(field,'new_reward_lr')
             pE.(field) = log(DCM.MDP.latent_lr/(1-DCM.MDP.latent_lr));             
             pC{i,i}    = 2;
         elseif strcmp(field,'latent_lr')
             pE.(field) = log(DCM.MDP.latent_lr/(1-DCM.MDP.latent_lr));             
             pC{i,i}    = 2;
         elseif strcmp(field,'new_latent_lr')
             pE.(field) = log(DCM.MDP.new_latent_lr/(1-DCM.MDP.new_latent_lr));             
             pC{i,i}    = 2;

        elseif strcmp(field,'decay')
             pE.(field) = log(DCM.MDP.decay/(1-DCM.MDP.decay));             
             pC{i,i}    = 2;
        elseif strcmp(field,'forget_threshold')
             pE.(field) = log(DCM.MDP.forget_threshold/(1-DCM.MDP.forget_threshold));             
             pC{i,i}    = 2;          
        elseif strcmp(field,'inverse_temp')
            pE.(field) = log(DCM.MDP.inverse_temp);             
            pC{i,i}    = 1;
        elseif strcmp(field,'reward_prior')
            pE.(field) = DCM.MDP.reward_prior;             
            pC{i,i}    = 1;  

        %%% DDM parameters    
        elseif any(strcmp(field,{'drift_baseline', 'drift'}))
            pE.(field) = DCM.MDP.(field);             
            pC{i,i}    = 0.5;  
        elseif any(strcmp(field,{'starting_bias', 'drift_mod', 'bias_mod'}))
            pE.(field) = log(DCM.MDP.(field)/(1-DCM.MDP.(field)));        
            pC{i,i}    = 0.1;  
        elseif any(strcmp(field,{'decision_thresh'}))
            pE.(field) = log(DCM.MDP.(field));             
            pC{i,i}    = 1;
        elseif any(strcmp(field,{'nondecision_time'}))
            pE.(field) =  -log((0.3 - 0.1) ./ (DCM.MDP.(field) - 0.1) - 1);             
            pC{i,i}    = 0.5;        
        else
            pE.(field) = 0;      
            pC{i,i}    = prior_variance;
        end
    end
end

pC      = spm_cat(pC);

% model specification
%--------------------------------------------------------------------------
M.L     = @(P,M,U,Y)spm_mdp_L(P,M,U,Y);  % log-likelihood function
M.pE    = pE;                            % prior means (parameters)
M.pC    = pC;                            % prior variance (parameters)
M.model = DCM.model;
M.decay_type = DCM.decay_type;
M.DCM = DCM;


% Variational Laplace
%--------------------------------------------------------------------------
[Ep,Cp,F] = spm_nlsi_Newton(M,DCM.U,DCM.Y);

% Store posterior densities and log evidnce (free energy)
%--------------------------------------------------------------------------
DCM.M   = M;
DCM.Ep  = Ep;
DCM.Cp  = Cp;
DCM.F   = F;


return
end

function [L] = spm_mdp_L(P,M,U,Y)
% log-likelihood function
% FORMAT L = spm_mdp_L(P,M,U,Y)
% P    - parameter structure
% M    - generative model
% U    - inputs
% Y    - observed repsonses
%__________________________________________________________________________

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% multiply parameters in MDP
%--------------------------------------------------------------------------
% mdp   = M.mdp;

field = fieldnames(M.pE);
for i = 1:length(field)
    if strcmp(field{i},'reward_lr')
        params.(field{i}) = 1/(1+exp(-P.(field{i}))); 
     elseif strcmp(field{i},'new_reward_lr')
         params.(field{i}) = 1/(1+exp(-P.(field{i})));   
     elseif strcmp(field{i},'latent_lr')
         params.(field{i}) = 1/(1+exp(-P.(field{i})));        
     elseif strcmp(field{i},'new_latent_lr')
         params.(field{i}) = 1/(1+exp(-P.(field{i}))); 
     elseif strcmp(field{i},'decay')
         params.(field{i}) = 1/(1+exp(-P.(field{i}))); 
    elseif strcmp(field{i},'forget_threshold')
         params.(field{i}) = 1/(1+exp(-P.(field{i}))); 
    elseif strcmp(field{i},'inverse_temp')
        params.(field{i}) = exp(P.(field{i}));           
    elseif strcmp(field{i},'reward_prior')
        params.(field{i}) = P.(field{i});
    
    %%% DDM Parameters
    elseif any(strcmp(field{i},{'drift_baseline', 'drift'}))
        params.(field{i}) = P.(field{i});
    elseif any(strcmp(field{i},{'starting_bias', 'drift_mod', 'bias_mod'}))
        params.(field{i}) = 1/(1+exp(-P.(field{i}))); 
    elseif any(strcmp(field{i},{'decision_thresh'}))
        params.(field{i}) = exp(P.(field{i}));
    elseif any(strcmp(field{i},{'nondecision_time'}))
        params.(field{i}) = 0.1 + (0.3 - 0.1) ./ (1 + exp(-P.(field{i})));     
    else
        mdp.(field{i}) = exp(P.(field{i}));
    end
end


trials = U;
L = 0;
decay_type = M.decay_type;
model_output = M.model(params, trials, decay_type, M.DCM); 



if M.DCM.sim == true
    trials = model_output.simmed_choices;
end

action_probabilities = model_output.patch_action_probs;
count = 0;
average_accuracy = 0;
average_action_probability = 0;
accuracy_count = 0;
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
action_accuracy = average_action_probability/count;
accuracy = average_accuracy/accuracy_count;

% If fitting a DDM, consider the pdf of dot motion reaction times in log
% likelihood
if M.DCM.use_DDM
    rt_pdf = model_output.dot_motion_rt_pdf;
    all_values = rt_pdf(~isnan(rt_pdf(:)));
    L = L + sum(log(all_values + eps));
end
    

 fprintf('LL: %f \n',L)
 fprintf('Average Patch Choice Probability: %f \n',action_accuracy)
 fprintf('Average Patch Choice Accuracy: %f \n',accuracy)
end





