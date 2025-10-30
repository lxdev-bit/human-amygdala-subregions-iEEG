function std_struct = decode_time_windows(data1, data2, ind, T, win_len, win_step, ...
                                          n_iter, train_ratio, seed, ...
                                          svm_s, svm_t, svm_c, svm_g)
% Sliding-window SVM decoding on z-power (e.g., cond1 vs. cond2), concise & anonymized.
%
% Deps:
%   - Deep Learning Toolbox: dividerand, mapminmax
%   - Statistics & Machine Learning Toolbox: pca
%   - LIBSVM: svmtrain, svmpredict
%
% Inputs:
%   data1, data2 : trials x channels x time
%   ind          : channel indices (e.g., 4:12)
%   T            : time index range (e.g., 1:4600)
%   win_len      : window length in samples (e.g., 100)
%   win_step     : window step in samples (e.g., 10)
%   n_iter       : iterations per window (e.g., 100)
%   train_ratio  : per-class train ratio (e.g., 0.7)
%   seed         : RNG seed (e.g., 2025)
%   svm_s,svm_t,svm_c,svm_g : LIBSVM params (e.g., 0,2,1,0.001)
%
% Output fields (std_struct):
%   predict_acc_ma [n_iter x n_win], best_acc_all [1 x n_win]
%   test_labels_all, predicted_labels_all, dec_values_all, test_indices_all (cells)
%   windows [2 x n_win] (start/stop in T), cfg (snapshot)

% ---- minimal defaults if args omitted/empty ----
if nargin < 13 || isempty(svm_g),  svm_g = 1e-3; end
if nargin < 12 || isempty(svm_c),  svm_c = 1;    end
if nargin < 11 || isempty(svm_t),  svm_t = 2;    end
if nargin < 10 || isempty(svm_s),  svm_s = 0;    end
if nargin < 9  || isempty(seed),   seed  = 2025; end
if nargin < 8  || isempty(train_ratio), train_ratio = 0.7; end
if nargin < 7  || isempty(n_iter), n_iter = 100; end
if nargin < 6  || isempty(win_step), win_step = 10; end
if nargin < 5  || isempty(win_len),  win_len  = 100; end
if nargin < 4  || isempty(T),        T = 1:size(data1,3); end
if nargin < 3  || isempty(ind),      ind = 1:size(data1,2); end

% ---- selection ----
data1 = data1(:, ind, T);
data2 = data2(:, ind, T);
mt = numel(T);

% windows: 2 x nWin (inclusive)
starts = 1:win_step:(mt - win_len + 1);
stops  = starts + win_len - 1;
mwn    = [starts; stops];
n_win  = size(mwn,2);

% ---- prealloc ----
predict_acc_ma        = zeros(n_iter, n_win);
best_acc_all          = zeros(1, n_win);
test_labels_all       = cell(n_iter, n_win);
predicted_labels_all  = cell(n_iter, n_win);
test_indices_all      = cell(n_iter, n_win);
dec_values_all        = cell(n_iter, n_win);

% ---- SVM params & RNG ----
param = sprintf('-s %d -t %d -c %g -g %g -q', svm_s, svm_t, svm_c, svm_g);
rng(seed);

% ---- main loop ----
for w = 1:n_win
    % slice one window
    X1 = data1(:, :, mwn(1,w):mwn(2,w));
    X2 = data2(:, :, mwn(1,w):mwn(2,w));

    % trials x (chan*time)
    A = reshape(X1, [size(X1,1), size(X1,2)*size(X1,3)]);
    B = reshape(X2, [size(X2,1), size(X2,2)*size(X2,3)]);

    acc_temp              = zeros(n_iter,1);
    test_labels_cell      = cell(n_iter,1);
    predicted_labels_cell = cell(n_iter,1);
    dec_values_cell       = cell(n_iter,1);
    test_indices_cell     = cell(n_iter,1);

    for it = 1:n_iter
        % stratified split per class (Deep Learning Toolbox)
        [a_tr, ~, a_te] = dividerand(size(A,1), train_ratio, 0, 1-train_ratio);
        [b_tr, ~, b_te] = dividerand(size(B,1), train_ratio, 0, 1-train_ratio);

        train_data  = [A(a_tr,:); B(b_tr,:)];
        test_data   = [A(a_te,:); B(b_te,:)];
        train_label = [ones(numel(a_tr),1); 2*ones(numel(b_tr),1)];
        test_label  = [ones(numel(a_te),1); 2*ones(numel(b_te),1)];

        % min-max on train, apply to test (Deep Learning Toolbox)
        [train_norm, PS] = mapminmax(train_data');  % cols=features
        train_norm = train_norm';
        test_norm  = mapminmax('apply', test_data', PS)'; 

        % PCA (99% variance), project test
        [pc, ~, latent] = pca(train_norm, 'Algorithm','svd', 'Centered', false);
        k = find(cumsum(latent)/sum(latent) >= 0.99, 1);
        if isempty(k), k = size(pc,2); end
        train_pca = train_norm * pc(:,1:k);
        test_pca  = test_norm  * pc(:,1:k);

        % SVM (LIBSVM)
        model = svmtrain(train_label, train_pca, param);
        [predicted_label, acc, dec_val] = svmpredict(test_label, test_pca, model);

        % store
        acc_temp(it)              = acc(1);
        test_labels_cell{it}      = test_label;
        predicted_labels_cell{it} = predicted_label;
        dec_values_cell{it}       = dec_val;
        test_indices_cell{it}     = [a_te(:); b_te(:) + size(A,1)]; % B offset
    end

    % per-window outputs
    predict_acc_ma(:,w)        = acc_temp;
    test_labels_all(:,w)       = test_labels_cell;
    predicted_labels_all(:,w)  = predicted_labels_cell;
    dec_values_all(:,w)        = dec_values_cell;
    test_indices_all(:,w)      = test_indices_cell;
    best_acc_all(w)            = mean(acc_temp);
end

% ---- pack ----
std_struct.predict_acc_ma        = predict_acc_ma;
std_struct.best_acc_all          = best_acc_all;
std_struct.test_labels_all       = test_labels_all;
std_struct.predicted_labels_all  = predicted_labels_all;
std_struct.dec_values_all        = dec_values_all;
std_struct.test_indices_all      = test_indices_all;
std_struct.windows               = mwn;
std_struct.cfg = struct('ind',ind,'T',T,'win_len',win_len,'win_step',win_step, ...
                        'n_iter',n_iter,'train_ratio',train_ratio,'seed',seed, ...
                        'svm_s',svm_s,'svm_t',svm_t,'svm_c',svm_c,'svm_g',svm_g);
end

