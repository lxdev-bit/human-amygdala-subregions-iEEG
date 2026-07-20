function shuffle_struct = conditional_decode_shuffle(data1, data2, ind, T, win_len, win_step, n_iter, train_ratio, n_shuffle, seed, svm_s, svm_t, svm_c, svm_g)

% Label-shuffling permutation analysis for sliding-window condition
% decoding using repeated train/test splits.
%
% Inputs:
%   data1, data2 : trials x channels x time
%   ind          : channel indices used for decoding
%   T            : time-sample indices used for decoding
%   win_len      : window length in samples
%   win_step     : window step in samples
%   n_iter       : repeated 70/30 splits for each permutation
%   train_ratio  : per-class training ratio, e.g., 0.70
%   n_shuffle    : number of label permutations
%   seed         : base random seed
%   svm_s        : LIBSVM SVM type
%   svm_t        : LIBSVM kernel type
%   svm_c        : LIBSVM cost parameter
%   svm_g        : LIBSVM gamma parameter
%
% Output fields:
%   shuffle_acc      : n_shuffle x n_win mean permutation accuracy
%   shuffle_acc_iter : n_shuffle x n_iter x n_win accuracy
%   windows          : 2 x n_win sliding-window indices
%   cfg              : analysis parameters

%% Combine trials

n_trial_1 = size(data1, 1);
n_trial_2 = size(data2, 1);
n_trial   = n_trial_1 + n_trial_2;

all_data = cat(1, data1, data2);

%% Initialize outputs

mt = numel(T);

starts = 1:win_step:(mt - win_len + 1);
stops  = starts + win_len - 1;
mwn    = [starts; stops];

n_win = size(mwn, 2);

shuffle_acc = zeros(n_shuffle, n_win);

shuffle_acc_iter = zeros( ...
    n_shuffle, ...
    n_iter, ...
    n_win);

%% Label permutations

for sh = 1:n_shuffle

    fprintf('Permutation %d / %d\n', sh, n_shuffle);

    % Randomly assign trials to two pseudoconditions while preserving
    % the original number of trials in each condition
    rng(seed + sh - 1);

    perm_order = randperm(n_trial);

    perm_data1 = all_data( ...
        perm_order(1:n_trial_1), ...
        :, ...
        :);

    perm_data2 = all_data( ...
        perm_order(n_trial_1 + 1:end), ...
        :, ...
        :);

    % Use a separate seed for the repeated 70/30 data splits
    split_seed = seed + 100000 + sh;

    % Run the original decoding function
    perm_result = conditional_decode_time_windows( ...
        perm_data1, ...
        perm_data2, ...
        ind, ...
        T, ...
        win_len, ...
        win_step, ...
        n_iter, ...
        train_ratio, ...
        split_seed, ...
        svm_s, ...
        svm_t, ...
        svm_c, ...
        svm_g);

    shuffle_acc(sh, :) = perm_result.best_acc_all;

    shuffle_acc_iter(sh, :, :) = perm_result.predict_acc_ma;
end

%% Pack outputs

shuffle_struct.shuffle_acc      = shuffle_acc;
shuffle_struct.shuffle_acc_iter = shuffle_acc_iter;
shuffle_struct.windows          = mwn;

shuffle_struct.cfg = struct( ...
    'ind', ind, ...
    'T', T, ...
    'win_len', win_len, ...
    'win_step', win_step, ...
    'n_iter', n_iter, ...
    'train_ratio', train_ratio, ...
    'n_shuffle', n_shuffle, ...
    'seed', seed, ...
    'svm_s', svm_s, ...
    'svm_t', svm_t, ...
    'svm_c', svm_c, ...
    'svm_g', svm_g);

end
