clc; clear;
load('E:\301-2506\results\neg-neu\1-tf-power\l_zpower_sig_chans.mat')
neg = permute(all_power_neg, [2,3,1]);
neu = permute(all_power_neu, [2,3,1]);
shape = permute(all_power_shape, [2,3,1]);

p_threshold = 0.05;
num_permutations = 1000;
frequencies = 1:180;

neg_stat = neg(:,501:4500,:);
neu_stat = neu(:,501:4500,:);
shape_stat = shape(:,501:4500,:);

[clusters_neg_neu, p_values_neg_neu, t_sums_neg_neu, ~] = permutest(neg_stat, neu_stat, true, p_threshold, num_permutations, true);
save('clus_neg_neu_l.mat', 'clusters_neg_neu', 'p_values_neg_neu', 't_sums_neg_neu');
[clusters_shape_neu, p_values_shape_neu, t_sums_shape_neu, ~] = permutest(neu_stat, shape_stat, true, p_threshold, num_permutations, true);
save('clus_shape_neu_l.mat', 'clusters_shape_neu', 'p_values_shape_neu', 't_sums_shape_neu');

%%
clc; clear;
load('E:\301-2506\results\neg-neu\1-tf-power\m_zpower_sig_chans.mat')
neg = permute(all_power_neg, [2,3,1]);
neu = permute(all_power_neu, [2,3,1]);
shape = permute(all_power_shape, [2,3,1]);

p_threshold = 0.05;
num_permutations = 1000;
frequencies = 1:180;

neg_stat = neg(:,501:4500,:);
neu_stat = neu(:,501:4500,:);
shape_stat = shape(:,501:4500,:);

[clusters_neg_neu, p_values_neg_neu, t_sums_neg_neu, ~] = permutest(neg_stat, neu_stat, true, p_threshold, num_permutations, true);
save('clus_neg_neu_m.mat', 'clusters_neg_neu', 'p_values_neg_neu', 't_sums_neg_neu');
[clusters_shape_neu, p_values_shape_neu, t_sums_shape_neu, ~] = permutest(neu_stat, shape_stat, true, p_threshold, num_permutations, true);
save('clus_shape_neu_m.mat', 'clusters_shape_neu', 'p_values_shape_neu', 't_sums_shape_neu');

%%
clc; clear;
load('E:\301-2506\results\neg-neu\1-tf-power\l_zpower_sig_chans.mat')
neg = permute(all_power_neg, [2,3,1]);
neu = permute(all_power_neu, [2,3,1]);
shape = permute(all_power_shape, [2,3,1]);

p_threshold = 0.01;
num_permutations = 10000;
frequencies = 1:180;

neg_stat = neg(:,501:4500,:);
neu_stat = neu(:,501:4500,:);
shape_stat = shape(:,501:4500,:);

[clusters_neg_neu, p_values_neg_neu, t_sums_neg_neu, ~] = permutest(neg_stat, neu_stat, true, p_threshold, num_permutations, true);
save('clus_neg_neu_l_10000.mat', 'clusters_neg_neu', 'p_values_neg_neu', 't_sums_neg_neu');
[clusters_shape_neu, p_values_shape_neu, t_sums_shape_neu, ~] = permutest(neu_stat, shape_stat, true, p_threshold, num_permutations, true);
save('clus_shape_neu_l_10000.mat', 'clusters_shape_neu', 'p_values_shape_neu', 't_sums_shape_neu');

%%
clc; clear;
load('E:\301-2506\results\neg-neu\1-tf-power\m_zpower_sig_chans.mat')
neg = permute(all_power_neg, [2,3,1]);
neu = permute(all_power_neu, [2,3,1]);
shape = permute(all_power_shape, [2,3,1]);

p_threshold = 0.01;
num_permutations = 10000;
frequencies = 1:180;

neg_stat = neg(:,501:4500,:);
neu_stat = neu(:,501:4500,:);
shape_stat = shape(:,501:4500,:);

[clusters_neg_neu, p_values_neg_neu, t_sums_neg_neu, ~] = permutest(neg_stat, neu_stat, true, p_threshold, num_permutations, true);
save('clus_neg_neu_m_10000.mat', 'clusters_neg_neu', 'p_values_neg_neu', 't_sums_neg_neu');
[clusters_shape_neu, p_values_shape_neu, t_sums_shape_neu, ~] = permutest(neu_stat, shape_stat, true, p_threshold, num_permutations, true);
save('clus_shape_neu_m_10000.mat', 'clusters_shape_neu', 'p_values_shape_neu', 't_sums_shape_neu');
