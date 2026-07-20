function [shuffle_acc, shuffle_acc_all, mwn] = subregional_decode_shuffle(all_data_1, all_data_2, ind, n_shuffle)
% SUBREGIONAL_DECODE_SHUFFLE
% Label-shuffling permutation analysis for time-resolved subregional
% decoding using leave-one-contact-out cross-validation.
%
% INPUTS
%   all_data_1 : nContact1 × nFreq × nTime data for subregion 1
%   all_data_2 : nContact2 × nFreq × nTime data for subregion 2
%   ind        : frequency indices used for decoding, e.g., 4:12
%   n_shuffle  : number of label permutations, e.g., 1000
%
% OUTPUTS
%   shuffle_acc
%       nShuffle × nWindow mean decoding accuracy.
%
%   shuffle_acc_all
%       nContact × nWindow × nShuffle fold-level accuracy.
%
%   mwn
%       Start and end sample indices of the sliding windows.

%% Combine contacts from the two subregions

n_sample_1 = size(all_data_1, 1);
n_sample_2 = size(all_data_2, 1);
n_sample   = n_sample_1 + n_sample_2;

all_data = cat(1, all_data_1, all_data_2);

all_label = [
    ones(n_sample_1, 1)
    2 * ones(n_sample_2, 1)
    ];

%% Initialize outputs

n_time = size(all_data, 3);

mwn(1,:) = 1:10:(n_time - 99);
mwn(2,:) = mwn(1,:) + 99;

n_window = size(mwn, 2);

shuffle_acc     = zeros(n_shuffle, n_window);
shuffle_acc_all = zeros(n_sample, n_window, n_shuffle);

%% Label-shuffling permutation

for sh = 1:n_shuffle

    fprintf('Shuffle %d / %d\n', sh, n_shuffle);

    % Randomly permute subregional labels while preserving class sizes
    shuffled_label = all_label(randperm(n_sample));

    shuffled_data_1 = all_data(shuffled_label == 1, :, :);
    shuffled_data_2 = all_data(shuffled_label == 2, :, :);

    % Run the same leave-one-contact-out decoding pipeline
    [acc_all, acc_mean, ~, ~, mwn] = ...
        subregional_decode_time_windows( ...
        shuffled_data_1, ...
        shuffled_data_2, ...
        ind);

    shuffle_acc(sh, :)       = acc_mean;
    shuffle_acc_all(:, :, sh) = acc_all;
end

end
