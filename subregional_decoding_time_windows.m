function [acc_all, acc_mean, pred_all, true_all, mwn, param] = ...
    decode_contact_loocv(all_data_1, all_data_2, ind)
% DECODE_CONTACT_LOOCV
% Sliding-window SVM decoding using leave-one-contact-out cross-validation.
%
% INPUTS
%   all_data_1 : nContact1 × nFreq × nTime
%   all_data_2 : nContact2 × nFreq × nTime
%   ind        : frequency indices used for decoding, e.g., 4:12
%
% OUTPUTS
%   acc_all  : fold × time-window accuracy
%   acc_mean : mean accuracy across folds
%   pred_all : predicted label for each fold
%   true_all : true label for each fold
%   mwn      : start and end indices of each time window
%   param    : LIBSVM parameter string

%% Sliding-window parameters

n_time = size(all_data_1, 3);

mwn(1,:) = 1:10:(n_time - 99);
mwn(2,:) = mwn(1,:) + 99;

%% LIBSVM parameters

s = 0;       % C-SVC
t = 0;       % linear kernel
c = 1;

param = sprintf('-s %d -t %d -c %g -q', s, t, c);

%% Sample labels

n_sample_1 = size(all_data_1, 1);
n_sample_2 = size(all_data_2, 1);
n_sample   = n_sample_1 + n_sample_2;

all_label = [
    ones(n_sample_1, 1)
    2 * ones(n_sample_2, 1)
    ];

%% Initialize outputs

n_window = size(mwn, 2);

acc_all  = zeros(n_sample, n_window);
acc_mean = zeros(1, n_window);

pred_all = zeros(n_sample, n_window);
true_all = repmat(all_label, 1, n_window);

%% Sliding-window decoding

for w = 1:n_window

    fprintf('Processing window %d / %d\n', w, n_window);

    % Extract current time-window data
    data_1_w = all_data_1(:, ind, mwn(1,w):mwn(2,w));
    data_2_w = all_data_2(:, ind, mwn(1,w):mwn(2,w));

    % contact × feature
    A = reshape(data_1_w, size(data_1_w,1), []);
    B = reshape(data_2_w, size(data_2_w,1), []);

    all_data = [A; B];

    acc_w  = zeros(n_sample, 1);
    pred_w = zeros(n_sample, 1);

    %% Leave-one-contact-out cross-validation

    for test_i = 1:n_sample

        train_idx = true(n_sample, 1);
        train_idx(test_i) = false;

        train_data  = all_data(train_idx, :);
        train_label = all_label(train_idx);

        test_data  = all_data(test_i, :);
        test_label = all_label(test_i);

        %% Normalize training data and apply to test data

        [train_norm, PS] = mapminmax(train_data');
        train_norm = train_norm';

        test_norm = mapminmax('apply', test_data', PS)';

        %% PCA fitted only to training data

        [coeff, ~, latent, ~, ~, mu] = pca( ...
            train_norm, ...
            'Algorithm', 'svd');

        explained_cum = cumsum(latent) ./ sum(latent);
        k = find(explained_cum >= 0.99, 1);

        train_pca = (train_norm - mu) * coeff(:, 1:k);
        test_pca  = (test_norm  - mu) * coeff(:, 1:k);

        %% Train and test SVM

        model = svmtrain(train_label, train_pca, param);

        pred_label = svmpredict( ...
            test_label, ...
            test_pca, ...
            model, ...
            '-q');

        pred_w(test_i) = pred_label;
        acc_w(test_i)  = double(pred_label == test_label) * 100;
    end

    acc_all(:, w)  = acc_w;
    pred_all(:, w) = pred_w;
    acc_mean(w)    = mean(acc_w);
end

end
