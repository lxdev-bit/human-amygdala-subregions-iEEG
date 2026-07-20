function loso_struct = conditional_LOSO_decode( ...
    data1_cell, data2_cell, ind, T, win_len, win_step, ...
    n_iter, n_shuffle, seed, ...
    svm_s, svm_t, svm_c, svm_g, pca_var_th)

% Subject-level leave-one-subject-out decoding with within-subject
% label-shuffling permutation analysis.
%
% Each cell contains all samples from one participant:
%   data1_cell{sub}: nSample1 x nFrequency x nTime
%   data2_cell{sub}: nSample2 x nFrequency x nTime
%
% INPUTS
%   data1_cell : subject-wise data for condition 1
%   data2_cell : subject-wise data for condition 2
%   ind        : frequency indices
%   T          : time-sample indices
%   win_len    : sliding-window length in samples
%   win_step   : sliding-window step in samples
%   n_iter     : repeated class-balancing iterations per LOSO fold
%   n_shuffle  : number of within-subject label permutations
%   seed       : random seed
%   svm_s      : LIBSVM SVM type
%   svm_t      : LIBSVM kernel type
%   svm_c      : LIBSVM cost parameter
%   svm_g      : LIBSVM gamma; use [] for 1/nPC
%   pca_var_th : retained PCA variance, e.g., 0.95
%
% MAIN OUTPUTS
%   real_acc_sub_iter  : nSubject x nWindow x nIteration raw accuracy
%   real_acc_sub_mean  : nSubject x nWindow accuracy averaged across iterations
%   real_bacc_sub_mean : nSubject x nWindow balanced accuracy
%   shuffle_acc_sub    : nSubject x nWindow x nShuffle accuracy
%   shuffle_bacc_sub   : nSubject x nWindow x nShuffle balanced accuracy
%   p_acc_pointwise, p_bacc_pointwise : pointwise empirical permutation p-values
%   p_acc_maxstat, p_bacc_maxstat     : maximum-statistic corrected permutation p-values

%% Select frequency and time samples
n_sub = numel(data1_cell);
for sub = 1:n_sub
    data1_cell{sub} = data1_cell{sub}(:, ind, T);
    data2_cell{sub} = data2_cell{sub}(:, ind, T);
end

%% Sliding windows
mt = numel(T);
starts = 1:win_step:(mt - win_len + 1);
stops  = starts + win_len - 1;
mwn   = [starts; stops];
n_win = size(mwn, 2);

%% Real LOSO decoding

[real_acc_sub_iter, real_bacc_sub_iter] = run_loso_decoding(data1_cell, data2_cell, mwn, n_iter, ...
    seed, svm_s, svm_t, svm_c, svm_g, pca_var_th);

real_acc_sub_mean = mean(real_acc_sub_iter, 3, 'omitnan');
real_bacc_sub_mean = mean(real_bacc_sub_iter, 3, 'omitnan');

%% Preallocate shuffle results
shuffle_acc_sub = nan(n_sub, n_win, n_shuffle);
shuffle_bacc_sub = nan(n_sub, n_win, n_shuffle);

%% Within-subject label permutations
for sh = 1:n_shuffle
    fprintf('Shuffle %d / %d\n', sh, n_shuffle);
    rng(seed + sh);
    perm_data1_cell = cell(n_sub, 1);
    perm_data2_cell = cell(n_sub, 1);
    for sub = 1:n_sub
        n1 = size(data1_cell{sub}, 1);
        n2 = size(data2_cell{sub}, 1);
        subject_data = cat(1, data1_cell{sub}, data2_cell{sub});
        perm_order = randperm(n1 + n2);
        perm_data1_cell{sub} = subject_data(perm_order(1:n1), :, :);
        perm_data2_cell{sub} = subject_data(perm_order(n1 + 1:end), :, :);
    end

    % Run the identical LOSO decoding pipeline on permuted labels
    [sh_acc_iter, sh_bacc_iter] = run_loso_decoding(perm_data1_cell, perm_data2_cell, mwn, ...
        n_iter, seed + 100000 + sh, svm_s, svm_t, svm_c, svm_g, pca_var_th);

    shuffle_acc_sub(:, :, sh) = mean(sh_acc_iter, 3, 'omitnan');
    shuffle_bacc_sub(:, :, sh) = mean(sh_bacc_iter, 3, 'omitnan');
end

%% Group-level real results

real_acc_mean = mean(real_acc_sub_mean, 1, 'omitnan');
real_acc_sem = std(real_acc_sub_mean, 0, 1, 'omitnan') ./ ...
    sqrt(sum(~isnan(real_acc_sub_mean), 1));
real_bacc_mean = mean(real_bacc_sub_mean, 1, 'omitnan');
real_bacc_sem = std(real_bacc_sub_mean, 0, 1, 'omitnan') ./ ...
    sqrt(sum(~isnan(real_bacc_sub_mean), 1));

%% Group-level permutation distributions

shuffle_acc_group = nan(n_shuffle, n_win);
shuffle_bacc_group = nan(n_shuffle, n_win);

for sh = 1:n_shuffle
    shuffle_acc_group(sh, :) = mean(shuffle_acc_sub(:, :, sh), 1, 'omitnan');
    shuffle_bacc_group(sh, :) = mean(shuffle_bacc_sub(:, :, sh), 1, 'omitnan');
end

%% Pointwise empirical permutation p-values
p_acc_pointwise = nan(1, n_win);
p_bacc_pointwise = nan(1, n_win);

for w = 1:n_win
    p_acc_pointwise(w) = ...
        (sum(shuffle_acc_group(:, w) >= real_acc_mean(w)) + 1) / ...
        (n_shuffle + 1);

    p_bacc_pointwise(w) = ...
        (sum(shuffle_bacc_group(:, w) >= real_bacc_mean(w)) + 1) / ...
        (n_shuffle + 1);
end

%% Maximum-statistic correction across time windows
max_acc_null = max(shuffle_acc_group, [], 2);
max_bacc_null = max(shuffle_bacc_group, [], 2);

p_acc_maxstat = nan(1, n_win);
p_bacc_maxstat = nan(1, n_win);

for w = 1:n_win
    p_acc_maxstat(w) = ...
        (sum(max_acc_null >= real_acc_mean(w)) + 1) / ...
        (n_shuffle + 1);

    p_bacc_maxstat(w) = ...
        (sum(max_bacc_null >= real_bacc_mean(w)) + 1) / ...
        (n_shuffle + 1);
end

thr95_acc_pointwise = prctile(shuffle_acc_group, 95, 1);
thr95_bacc_pointwise = prctile(shuffle_bacc_group, 95, 1);


%% Pack outputs

loso_struct.real_acc_sub_iter  = real_acc_sub_iter;
loso_struct.real_bacc_sub_iter = real_bacc_sub_iter;
loso_struct.real_acc_sub_mean  = real_acc_sub_mean;
loso_struct.real_bacc_sub_mean = real_bacc_sub_mean;
loso_struct.real_acc_mean  = real_acc_mean;
loso_struct.real_acc_sem   = real_acc_sem;
loso_struct.real_bacc_mean = real_bacc_mean;
loso_struct.real_bacc_sem  = real_bacc_sem;
loso_struct.shuffle_acc_sub  = shuffle_acc_sub;
loso_struct.shuffle_bacc_sub = shuffle_bacc_sub;
loso_struct.shuffle_acc_group  = shuffle_acc_group;
loso_struct.shuffle_bacc_group = shuffle_bacc_group;
loso_struct.p_acc_pointwise  = p_acc_pointwise;
loso_struct.p_bacc_pointwise = p_bacc_pointwise;
loso_struct.thr95_acc_pointwise  = thr95_acc_pointwise;
loso_struct.thr95_bacc_pointwise = thr95_bacc_pointwise;
loso_struct.windows = mwn;
loso_struct.window_sample_indices = [T(mwn(1, :))  T(mwn(2, :))];
loso_struct.cfg = struct('ind', ind, 'T', T, 'win_len', win_len, ...
    'win_step', win_step, 'n_iter', n_iter, 'n_shuffle', n_shuffle, ...
    'seed', seed, 'svm_s', svm_s, 'svm_t', svm_t, 'svm_c', svm_c, ...
    'svm_g', svm_g, 'pca_var_th', pca_var_th);

end


function [acc_sub_iter, bacc_sub_iter] = run_loso_decoding(data1_cell, data2_cell, mwn, n_iter, run_seed, svm_s, svm_t, svm_c, svm_g, pca_var_th)
% Identical LOSO pipeline used for real and shuffled data.

n_sub = numel(data1_cell);
n_win = size(mwn, 2);

acc_sub_iter = nan(n_sub, n_win, n_iter);
bacc_sub_iter = nan(n_sub, n_win, n_iter);

for test_sub = 1:n_sub
    fprintf('LOSO fold %d / %d\n', test_sub, n_sub);
    train_subs = setdiff(1:n_sub, test_sub);

    % Training sample numbers are constant across time windows
    n_train_1 = 0;
    n_train_2 = 0;

    for sub = train_subs
        n_train_1 = n_train_1 + size(data1_cell{sub}, 1);
        n_train_2 = n_train_2 + size(data2_cell{sub}, 1);
    end

    n_min = min(n_train_1, n_train_2);

    % Use the same class-balancing samples across all time windows
    rng(run_seed + test_sub);

    balance_idx_1 = zeros(n_iter, n_min);
    balance_idx_2 = zeros(n_iter, n_min);

    for it = 1:n_iter
        balance_idx_1(it, :) = randperm(n_train_1, n_min);
        balance_idx_2(it, :) = randperm(n_train_2, n_min);
    end

    for w = 1:n_win

        tidx = mwn(1, w):mwn(2, w);

        %% Held-out participant
        X_test_1 = extract_window_feature(data1_cell{test_sub}, tidx);
        X_test_2 = extract_window_feature(data2_cell{test_sub}, tidx);

        X_test = [X_test_1 X_test_2];
        y_test = [ones(size(X_test_1, 1), 1)  2 * ones(size(X_test_2, 1), 1)];

        %% All remaining participants
        X_train_1_all = [];
        X_train_2_all = [];

        for sub = train_subs
            X_train_1_all = [X_train_1_all extract_window_feature(data1_cell{sub}, tidx)];
            X_train_2_all = [X_train_2_all extract_window_feature(data2_cell{sub}, tidx)];
        end

        for it = 1:n_iter
            X_train_1 = X_train_1_all(balance_idx_1(it, :), :);
            X_train_2 = X_train_2_all(balance_idx_2(it, :), :);

            X_train = [X_train_1 X_train_2];
            y_train = [ones(n_min, 1) 2 * ones(n_min, 1)];

            %% Training-set normalization
            [X_train_norm, PS] = mapminmax(X_train');
            X_train_norm = X_train_norm';

            X_test_norm = mapminmax('apply', X_test', PS)';

            %% Remove constant training features
            valid_feature = std(X_train_norm, 0, 1) > 0;

            X_train_norm = X_train_norm(:, valid_feature);
            X_test_norm  = X_test_norm(:, valid_feature);

            %% Training-set PCA
            [pc, ~, latent, ~, ~, mu] = pca(X_train_norm, 'Algorithm', 'svd');
            explained_cumulative = cumsum(latent) ./ sum(latent);

            k = find(explained_cumulative >= pca_var_th, 1);

            X_train_pca = bsxfun(@minus, X_train_norm, mu) * pc(:, 1:k);
            X_test_pca = bsxfun(@minus, X_test_norm, mu) * pc(:, 1:k);

            %% LIBSVM
            param = sprintf('-s %d -t %d -c %g -q', svm_s, svm_t, svm_c);
            model = svmtrain(y_train, X_train_pca, param);
            predicted_label = svmpredict(y_test, X_test_pca, model, '-q');

            %% Accuracy
            acc_sub_iter(test_sub, w, it) = mean(predicted_label == y_test);
            acc_class_1 = mean(predicted_label(y_test == 1) == 1);
            acc_class_2 = mean(predicted_label(y_test == 2) == 2);
            bacc_sub_iter(test_sub, w, it) = mean([acc_class_1, acc_class_2]);
        end
    end
end

end


function X = extract_window_feature(data, tidx)
% DATA: nSample x nFrequency x nTime
% X:    nSample x (nFrequency*nWindowTime)

tmp = data(:, :, tidx);

X = reshape( ...
    tmp, ...
    size(tmp, 1), ...
    size(tmp, 2) * size(tmp, 3));

end
