function result = subregional_LOSO_decode(data_l, data_m, subid_l, subid_m, ...
    freq_ind, time_ind, n_shuffle, seed, svm_s, svm_t, svm_c, svm_g, pca_var_th)
% 
%
% Fixed-window lateral-versus-medial amygdala decoding using
% leave-one-subject-out cross-validation and within-subject label
% permutation.
%
% INPUTS
%   data_l      : nLateralContact × nFrequency × nTime
%   data_m      : nMedialContact  × nFrequency × nTime
%   subid_l     : subject ID for each lateral contact
%   subid_m     : subject ID for each medial contact
%   freq_ind    : frequency indices, e.g., 4:12
%   time_ind    : fixed time-sample indices
%   n_shuffle   : number of label permutations
%   seed        : random seed
%   svm_s       : LIBSVM SVM type, usually 0
%   svm_t       : LIBSVM kernel type, 0 = linear, 2 = RBF
%   svm_c       : LIBSVM cost parameter
%   svm_g       : LIBSVM gamma; ignored for linear kernel
%   pca_var_th  : retained PCA variance, e.g., 0.99
%
% OUTPUT
%   result.real_acc_sub : Subject-level overall decoding accuracy.
%   result.real_bacc_sub : Subject-level balanced accuracy.
%   result.shuffle_acc_sub : nSubject × nShuffle overall accuracy null distribution.
%   result.shuffle_bacc_sub : nSubject × nShuffle balanced accuracy null distribution.
%   result.real_acc_group : Mean accuracy across held-out subjects.
%   result.real_bacc_group :  Mean balanced accuracy across held-out subjects.
%   result.shuffle_acc_group : Group mean accuracy for each permutation.
%   result.shuffle_bacc_group : Group mean balanced accuracy for each permutation.
%   result.p_acc_group, result.p_bacc_group : Group-level empirical permutation p-values.

%% Extract fixed frequency and time range
data_l = data_l(:, freq_ind, time_ind);
data_m = data_m(:, freq_ind, time_ind);

% contact × (frequency × time)
X_l = reshape(data_l, size(data_l, 1), size(data_l, 2) * size(data_l, 3));
X_m = reshape(data_m, size(data_m, 1), size(data_m, 2) * size(data_m, 3));

%% Combine contacts
X = [X_l X_m];
label = [ones(size(X_l, 1), 1)  2 * ones(size(X_m, 1), 1)];
subject_id = [subid_l(:)  subid_m(:)];

% Only subjects containing contacts from both subregions are used as
% held-out test subjects.
test_subjects = intersect(unique(subid_l(:)), unique(subid_m(:)));
n_sub = numel(test_subjects);

%% LIBSVM parameters
param = sprintf('-s %d -t %d -c %g -g %g -q', svm_s, svm_t, svm_c, svm_g);

%% Real LOSO decoding
[real_acc_sub, real_bacc_sub, real_predicted_labels, real_test_labels] = run_fixed_window_loso( ...
        X, label, subject_id, test_subjects, param, pca_var_th);

%% Shuffle distribution
shuffle_acc_sub = nan(n_sub, n_shuffle);
shuffle_bacc_sub = nan(n_sub, n_shuffle);
all_subjects = unique(subject_id);

for sh = 1:n_shuffle

    fprintf('Shuffle %d / %d\n', sh, n_shuffle);
    rng(seed + sh - 1);
    shuffled_label = label;

    % Shuffle lateral/medial labels separately within each participant.
    % The original number of lateral and medial contacts is preserved.
    for s = 1:numel(all_subjects)
        current_subject = all_subjects(s);
        subject_idx = find(subject_id == current_subject);
        shuffled_label(subject_idx) = label(subject_idx(randperm(numel(subject_idx))));
    end

    [sh_acc, sh_bacc] = run_fixed_window_loso(X, shuffled_label, ...
        subject_id, test_subjects, param, pca_var_th);

    shuffle_acc_sub(:, sh) = sh_acc;
    shuffle_bacc_sub(:, sh) = sh_bacc;
end

%% Group-level results
real_acc_group = mean(real_acc_sub, 'omitnan');
real_bacc_group = mean(real_bacc_sub, 'omitnan');

shuffle_acc_group = mean(shuffle_acc_sub, 1, 'omitnan');
shuffle_bacc_group = mean(shuffle_bacc_sub, 1, 'omitnan');

p_acc_group = (sum(shuffle_acc_group >= real_acc_group) + 1) / (n_shuffle + 1);
p_bacc_group = (sum(shuffle_bacc_group >= real_bacc_group) + 1) / (n_shuffle + 1);

%% Subject-level empirical p-values
p_acc_sub = nan(n_sub, 1);
p_bacc_sub = nan(n_sub, 1);

for s = 1:n_sub
    p_acc_sub(s) = (sum(shuffle_acc_sub(s, :) >= real_acc_sub(s)) + 1) / (n_shuffle + 1);
    p_bacc_sub(s) = (sum(shuffle_bacc_sub(s, :) >= real_bacc_sub(s)) + 1) / (n_shuffle + 1);
end

%% Pack outputs

result.test_subjects = test_subjects;
result.real_acc_sub = real_acc_sub;
result.real_bacc_sub = real_bacc_sub;
result.real_acc_group = real_acc_group;
result.real_bacc_group = real_bacc_group;
result.shuffle_acc_sub = shuffle_acc_sub;
result.shuffle_bacc_sub = shuffle_bacc_sub;
result.shuffle_acc_group = shuffle_acc_group;
result.shuffle_bacc_group = shuffle_bacc_group;
result.p_acc_sub = p_acc_sub;
result.p_bacc_sub = p_bacc_sub;
result.p_acc_group = p_acc_group;
result.p_bacc_group = p_bacc_group;
result.real_predicted_labels = real_predicted_labels;
result.real_test_labels = real_test_labels;

result.cfg = struct('freq_ind', freq_ind, 'time_ind', time_ind, 'n_shuffle', n_shuffle, ...
    'seed', seed, 'svm_s', svm_s, 'svm_t', svm_t, 'svm_c', svm_c,'svm_g', svm_g, ...
    'pca_var_th', pca_var_th, 'param', param);

end


function [acc_sub, bacc_sub, predicted_all, true_all] = run_fixed_window_loso( X, label, subject_id, test_subjects, param, pca_var_th)
%
% Run one complete subject-level LOSO analysis.

n_sub = numel(test_subjects);

acc_sub = nan(n_sub, 1);
bacc_sub = nan(n_sub, 1);

predicted_all = cell(n_sub, 1);
true_all = cell(n_sub, 1);

for test_s = 1:n_sub
    current_subject = test_subjects(test_s);
    fprintf('LOSO subject %d / %d: %g\n', test_s, n_sub, current_subject);

    % All contacts from the held-out participant enter the test set.
    test_idx = subject_id == current_subject;

    % No contact from the held-out participant enters the training set.
    train_idx = subject_id ~= current_subject;
    train_data = X(train_idx, :);
    train_label = label(train_idx);
    test_data = X(test_idx, :);
    test_label = label(test_idx);

    %% Normalize using training data only
    [train_norm, PS] = mapminmax(train_data');
    train_norm = train_norm';
    test_norm = mapminmax('apply', 'test_data', PS)';

    %% PCA using training data only
    [coeff, ~, latent, ~, ~, mu] = pca(train_norm, 'Algorithm', 'svd');
    explained_cum = cumsum(latent) ./ sum(latent);
    k = find(explained_cum >= pca_var_th, 1);

    train_pca = (train_norm - mu) * coeff(:, 1:k);
    test_pca = (test_norm - mu) * coeff(:, 1:k);

    %% SVM
    model = svmtrain(train_label, train_pca, param);
    predicted_label = svmpredict(test_label, test_pca, model, '-q');

    %% Subject-level accuracy
    acc_sub(test_s) = mean(predicted_label == test_label) * 100;
    lateral_acc = mean(predicted_label(test_label == 1) == 1);
    medial_acc = mean(predicted_label(test_label == 2) == 2);
    bacc_sub(test_s) =  mean([lateral_acc, medial_acc]) * 100;

    predicted_all{test_s} = predicted_label;
    true_all{test_s} = test_label;
end

end
