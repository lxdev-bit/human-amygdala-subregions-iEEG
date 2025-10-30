function [gc, f] = cal_npCGC(X, fs, fRes)
% cal_npCGC
% Calculate frequency-domain conditional Granger causality (nonparametric)
% from multichannel data using multitaper spectra + Wilson factorization.
%
% Inputs
%   X   : [time x trial x channel]
%   fs  : sampling rate (Hz)
%   fRes: desired frequency resolution (Hz). If empty or > fs/Nt, uses fs/Nt
%
% Outputs
%   gc  : [F x C x C] conditional GC (j -> i is gc(:,i,j)), F=#freqs, C=#channels
%   f   : [1 x F] frequency vector (Hz)
%
% Core steps adapted from your original compute_allnpCGCvar3 (keeping only M.gc path):
%   1) Multitaper cross-spectra (sig2mTspect_nv)
%   2) Wilson spectral factorization (wilson_sf)
%   3) Conditional GC for all pairs (hz2cgcAll)
%
% Ref: M. Dhamala et al., NeuroImage (2008).

% ------------ spectra ------------
[Nt, ~, ~] = size(X);
if nargin < 3 || isempty(fRes) || fRes > fs/Nt
    fRes = fs/Nt; % default resolution via zero padding policy in original code
end
[S, f] = sig2mTspect_nv_local(X, fs, fRes);   % S: [m x m x F]

% ------------ Wilson factorization ------------
[H, Z] = wilson_sf_local(S, fs);               % H: [m x m x F], Z: [m x m]

% ------------ conditional GC ------------
gc = hz2cgcAll_local(S, H, Z, f, fs);          % [F x m x m]
end

% ======================================================================
% ======== Local helper implementations (trimmed to GC-only path) ======
% ======================================================================

function [S,f]= sig2mTspect_nv_local(X,fs,fRes)
% Multitaper auto/cross spectra (time x trial x chan) -> S: [chan x chan x F]
[N,Ntr,m] = size(X);
if nargin<3 || fRes>fs/N
    npad = 0;  fRes = fs/N;
end
fRes0 = fs/N;
if ~isempty(fRes) && (fRes <= fs/N)
    npad = round((fs/fRes - N)/2);  % zeros padded on each side
end
f = fs*(0:fix((N+2*npad)/2))/(N+2*npad);  % 0..Nyquist

nw = 2; % number of DPSS tapers = 2*nw
[tapers,~] = dpss(N+2*npad, nw);

S = zeros(m,m,N+2*npad);
for itrial = 1:Ntr
    Xft = zeros(N+2*npad, size(tapers,2), m);
    for ii = 1:m
        Xft(:,:,ii) = mtfft_local(squeeze(X(:,itrial,ii)), tapers, npad);
    end
    s = zeros(m,m,N+2*npad);
    for ii = 1:m
        for jj = 1:m
            s(ii,jj,:) = squeeze(mean(Xft(:,:,ii).*conj(Xft(:,:,jj)),2)); % avg over tapers
        end
    end
    S = S + s;
end
S = S/Ntr;                                      % avg over trials
S = S(:,:,1:fix(end/2)+1)/fs;                   % one-sided spectra
S = S*(fRes0/fRes);                             % adjust for zero padding
end

function xf = mtfft_local(data,tapers,npad)
x0  = zeros(npad,1);
xez = [x0; data; x0];
xmt = xez .* tapers;                            % apply tapers (columns)
xf  = fft(xmt, [], 1);
end

% ---------------- Wilson spectral factorization (trimmed) ----------------
function [H, Z, ps, ps0, converged, relerr] = wilson_sf_local(S, fs, tol)
if (nargin < 3) || isempty(tol), tol = 1e-9; end
[k, ~, N] = size(S);

Sarr = cat(3, S, conj(S(:,:,N-1:-1:2)));
ps0 = ps0_initial__(Sarr);
ps  = repmat(ps0, [1,1,N]);
ps  = cat(3, ps, conj(ps(:,:,N-1:-1:2)));
M   = size(Sarr, 3);

I = eye(k);
maxiter = min(500, floor(sqrt(10/tol)));

U = zeros(size(Sarr));
for j = 1:M
    U(:,:,j) = chol(Sarr(:,:,j));
end

niter = 0; converged = false;
g = zeros(k,k,M);
while (niter < maxiter) && ~converged
    for i = 1:M
        V = ps(:,:,i)\U(:,:,i)';                 % ps^{-1} * U^T
        g(:,:,i) = V*V' + I;
    end
    [gp, gp0] = PlusOperator__(g);
    T = -tril(gp0, -1); T = T - T';

    ps_prev = ps;
    for i = 1:M
        ps(:,:,i) = ps(:,:,i)*(gp(:,:,i) + T);
    end
    ps0_prev = ps0; ps0 = ps0*(gp0 + T);

    [converged, relerr] = check_converged_ps__(ps, ps_prev, ps0, ps0_prev, tol); 
    niter = niter + 1;
end

H = zeros(k,k,N);
for i = 1:N
    H(:,:,i) = ps(:,:,i)/ps0;
end
ps  = sqrt(fs)*ps(:,:,1:N); 
ps0 = sqrt(fs)*ps0;
Z   = ps0*ps0';
end

function ps0 = ps0_initial__(Sarr)
[k, ~, M] = size(Sarr);
Sarr  = reshape(Sarr, [k*k, M]);
gamma = ifft(transpose(Sarr));
gamma0 = reshape(gamma(1,:), [k k]);
gamma0 = real((gamma0 + gamma0')/2);
ps0 = chol(gamma0);
end

function [gp, gp0] = PlusOperator__(g)
[k, ~, M] = size(g);
N = ceil((M+1)/2);
g  = reshape(g,[k*k, M]);
G  = real(ifft(transpose(g)));
G  = reshape(transpose(G),[k,k,M]);
G(:,:,1) = 0.5*G(:,:,1); gp0 = G(:,:,1);
G(:,:,N+1:end) = 0;
G  = reshape(G,[k*k, M]);
gp = fft(transpose(G));
gp = reshape(transpose(gp),[k,k,M]);
end

function [ok, relerr] = check_converged_ps__(ps, ps_prev, ps0, ps0_prev, tol)
[ok, relerr] = CheckRelErr__(ps0, ps0_prev, tol);
if ok
    [ok2, rel2] = CheckRelErr__(ps, ps_prev, tol);
    ok = ok2; relerr = max(relerr, rel2);
end
end

function [ok, relerr] = CheckRelErr__(A,B,reltol)
D = abs(B - A);
A2 = abs(A); A2(A2 <= 2*eps) = 1;
E = D ./ A2;
relerr = max(E(:));
ok = (relerr <= reltol);
end

% ---------------- Conditional GC for all pairs ----------------
function cgc = hz2cgcAll_local(S,H,Z,freq,fs) %#ok<INUSD>
nc  = size(H,1);
F   = numel(freq);
cgc = zeros(F, nc, nc);
for i = 1:nc
    for j = 1:nc
        if i==j, continue; end
        [Zij,Zijk,Zkij,Zkk] = pttmatrx_local(Z,i,j);
        Z3=[Zij,Zijk;Zkij,Zkk];
        cc=Z3(1,1)*Z3(2,2)-Z3(1,2)^2;
        P=[1,0,zeros(1,nc-2); ...
           -Z3(1,2)/Z3(1,1),1,zeros(1,nc-2); ...
           (Z3(1,2)*Z3(2,3:end)-Z3(2,2)*Z3(1,3:end))'./cc, ...
           (Z3(1,2)*Z3(1,3:end)-Z3(1,1)*Z3(2,3:end))'./cc, eye(nc-2)];

        % build S2, H3 along freq
        S2 = zeros(2,2,F); H3 = zeros(2,2,F);
        for fi = 1:F
            [Sij,Sijk,Skij,Skk] = pttmatrx_local(S(:,:,fi),i,j);
            [Hij,Hijk,Hkij,Hkk] = pttmatrx_local(H(:,:,fi),i,j);
            S2(:,:,fi) = [Sij(1,1),Sijk(1,:); Skij(:,1), Skk];
            H3(:,:,fi) = [Hij, Hijk; Hkij, Hkk];
        end
        [H2,Z2] = wilson_sf_local(S2, 1);      % normalization factor; 1 is sufficient
        for fi = 1:F
            HH = squeeze(H3(:,:,fi))/P;
            Q  = [1,zeros(1,nc-2); -Z2(1,2:end)'./Z2(1,1), eye(nc-2)];
            B  = Q / squeeze(H2(:,:,fi));
            BB = [B(1,1),0,B(1,2:end); 0,1,zeros(1,nc-2); B(2:end,1),zeros(1,nc-2),B(2:end,2:end)];
            FF = BB*HH;
            cgc(fi,j,i) = log(abs(Z2(1,1))/abs(FF(1,1)*Z3(1,1)*conj(FF(1,1))));
        end
    end
end
end

function [B11,B12,B21,B22]=pttmatrx_local(H,i,j)
B1 = H; B2 = H;
B11 = [H(i,i),H(j,i); H(i,j),H(j,j)];
B1(i,:) = [];
if j<i, B1(j,:) = []; else, B1(j-1,:) = []; end
B21 = [B1(:,i), B1(:,j)];
B1(:,i) = [];
if j<i, B1(:,j) = []; else, B1(:,j-1) = []; end
B2(:,i) = [];
if j<i, B2(:,j) = []; else, B2(:,j-1) = []; end
B12 = [B2(i,:); B2(j,:)]; 
B22 = B1;                  
end
