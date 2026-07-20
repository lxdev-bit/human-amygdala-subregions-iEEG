function result = tf_curve_stat(data_neg, data_neu, data_shape, freq_ind, times, p_threshold, num_permutations)

% Extract frequency-averaged time courses and perform paired cluster-based
% permutation tests between conditions.
%
% Inputs:
%   data_neg, data_neu, data_shape : observation × frequency × time
%   freq_ind                       : selected frequency indices, e.g. 4:12
%   times                          : time vector corresponding to dimension 3
%   p_threshold                    : cluster-forming threshold, e.g. 0.05
%   num_permutations               : number of permutations, e.g. 1000
%
% Outputs:
%   result.band_data   : observation-level frequency-averaged curves
%   result.mean        : condition-level mean curves
%   result.sem         : standard error curves
%   result.stats       : cluster permutation results
%   result.cfg         : analysis parameters

%% Average across selected frequencies

neg = mean(data_neg(:, freq_ind, :), 2, 'omitnan');
neu = mean(data_neu(:, freq_ind, :), 2, 'omitnan');
shape = mean(data_shape(:, freq_ind, :), 2, 'omitnan');

thal_neg = reshape(neg, size(data_neg,1), []);
thal_neu = reshape(neu, size(data_neu,1), []);
thal_shape = reshape(shape, size(data_shape,1), []);

%% Mean and SEM

mean_neg = mean(thal_neg, 1, 'omitnan');
mean_neu = mean(thal_neu, 1, 'omitnan');
mean_shape = mean(thal_shape, 1, 'omitnan');

sem_neg = std(thal_neg, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(thal_neg), 1));
sem_neu = std(thal_neu, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(thal_neu), 1));
sem_shape = std(thal_shape, 0, 1, 'omitnan') ./ sqrt(sum(~isnan(thal_shape), 1));

%% Paired cluster-based permutation tests

[clusters_neg_neu, p_neg_neu, t_neg_neu, null_neg_neu] = ...
    permutest(thal_neg', thal_neu', true, p_threshold, num_permutations, true);

[clusters_neu_shape, p_neu_shape, t_neu_shape, null_neu_shape] = ...
    permutest(thal_neu', thal_shape', true, p_threshold, num_permutations, true);

[clusters_neg_shape, p_neg_shape, t_neg_shape, null_neg_shape] = ...
    permutest(thal_neg', thal_shape', true, p_threshold, num_permutations, true);

%% Pack results

result.times = times;

result.band_data.neg = thal_neg;
result.band_data.neu = thal_neu;
result.band_data.shape = thal_shape;

result.mean.neg = mean_neg;
result.mean.neu = mean_neu;
result.mean.shape = mean_shape;

result.sem.neg = sem_neg;
result.sem.neu = sem_neu;
result.sem.shape = sem_shape;

result.stats.neg_neu.clusters = clusters_neg_neu;
result.stats.neg_neu.p_values = p_neg_neu;
result.stats.neg_neu.t_sums = t_neg_neu;
result.stats.neg_neu.null_distribution = null_neg_neu;

result.stats.neu_shape.clusters = clusters_neu_shape;
result.stats.neu_shape.p_values = p_neu_shape;
result.stats.neu_shape.t_sums = t_neu_shape;
result.stats.neu_shape.null_distribution = null_neu_shape;

result.stats.neg_shape.clusters = clusters_neg_shape;
result.stats.neg_shape.p_values = p_neg_shape;
result.stats.neg_shape.t_sums = t_neg_shape;
result.stats.neg_shape.null_distribution = null_neg_shape;

result.cfg.freq_ind = freq_ind;
result.cfg.p_threshold = p_threshold;
result.cfg.num_permutations = num_permutations;
result.cfg.paired = true;

end
