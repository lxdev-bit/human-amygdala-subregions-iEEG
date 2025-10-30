function dwpli_struct = cal_dwpli(EEG, channel1, channel2, freqs2use, baselinetm, min_cycles, max_cycles)
% cal_dwpli
% Calculate debiased weighted Phase Lag Index (dwPLI) between two channels
% across time and frequencies using Morlet wavelets (convolution in FFT domain).
%
% Reference/implementation adapted from:
% https://github.com/mikexcohen/AnalyzingNeuralTimeSeries/blob/main/chapter26.m
%
% Inputs
%   EEG         : EEGLAB struct (fields: data [chan x pnts x trials], srate, times, chanlocs)
%   channel1    : char, e.g., 'Fz' (case-insensitive)
%   channel2    : char, e.g., 'O1' (case-insensitive)
%   freqs2use   : vector of frequencies to analyze (default 1:150)
%   baselinetm  : [t1 t2] ms, baseline window on EEG.times (default [-400 -200])
%   min_cycles  : min #cycles at lowest frequency (default 4)
%   max_cycles  : max #cycles at highest frequency (default 12)
%
% Output
%   out struct with fields:
%     .dwpli         : [nfreq x nt] baseline-corrected dwPLI
%     .dwpli_raw     : [nfreq x nt] raw dwPLI before baseline subtraction
%     .freqs         : freqs2use
%     .times         : EEG.times (ms)
%     .baseline_ms   : baselinetm
%     .channels      : {channel1, channel2}
%     .params        : struct of wavelet/FFT params
%
% Notes
%   - This function keeps the original full temporal resolution (EEG.pnts).
%   - No file I/O; no subject/condition identifiers; concise by design.

% -------- defaults --------
if nargin < 4 || isempty(freqs2use),  freqs2use  = 1:150; end
if nargin < 5 || isempty(baselinetm), baselinetm = [-400 -200]; end
if nargin < 6 || isempty(min_cycles), min_cycles = 4; end
if nargin < 7 || isempty(max_cycles), max_cycles = 12; end

% -------- basic sizes/indices --------
srate = EEG.srate;
nt    = EEG.pnts;
ntr   = EEG.trials;
times = EEG.times(:)';

% channel indices
chanidx = zeros(1,2);
chanidx(1) = find(strcmpi(channel1, {EEG.chanlocs.labels}), 1, 'first');
chanidx(2) = find(strcmpi(channel2, {EEG.chanlocs.labels}), 1, 'first');
assert(all(~isnan(chanidx)) && all(chanidx>0), 'Channel(s) not found in EEG.chanlocs.');

% baseline (full-resolution) indices
baselineidxF = dsearchn(times', baselinetm(:));
baselineidxF = baselineidxF(:)';

% -------- wavelet & FFT params --------
tvec          = -1:1/srate:1;                   % seconds
half_wavelet  = (numel(tvec)-1)/2;
n_wavelet     = numel(tvec);

% cycles per frequency: interpolate between low/high freqs
num_cycles = interp1([min(freqs2use) max(freqs2use)], ...
                     [min_cycles      max_cycles     ], ...
                     freqs2use, 'pchip');

% convolution sizes
n_data        = nt * ntr;
n_convolution = n_wavelet + n_data - 1;

% -------- pack continuous data & FFT once per channel --------
data_fft1 = fft(reshape(double(EEG.data(chanidx(1),:,:)), 1, n_data), n_convolution);
data_fft2 = fft(reshape(double(EEG.data(chanidx(2),:,:)), 1, n_data), n_convolution);

% -------- init outputs --------
nfreq = numel(freqs2use);
dwpli = zeros(nfreq, nt);

% -------- main frequency loop --------
for fi = 1:nfreq
    f = freqs2use(fi);

    % Morlet wavelet in time and its FFT
    s  = num_cycles(fi) / (2*pi*f);
    wave = exp(2*1i*pi*f.*tvec) .* exp(-tvec.^2 ./ (2*s^2));
    wave_fft = fft(wave, n_convolution);

    % convolution -> analytic signals for two channels
    conv1 = ifft(wave_fft .* data_fft1, n_convolution);
    conv1 = conv1(half_wavelet+1:end-half_wavelet);    % trim
    sig1  = reshape(conv1, nt, ntr);                   % time x trials

    conv2 = ifft(wave_fft .* data_fft2, n_convolution);
    conv2 = conv2(half_wavelet+1:end-half_wavelet);
    sig2  = reshape(conv2, nt, ntr);

    % cross-spectral density & imaginary part
    cdd = sig1 .* conj(sig2);                          % time x trials
    cdi = imag(cdd);

    % debiased weighted phase-lag index (FieldTrip-style shortcut)
    imagsum      = sum(cdi,  2);
    imagsumW     = sum(abs(cdi), 2);
    debiasfactor = sum(cdi.^2, 2);
    dwpli(fi,:)  = ((imagsum.^2 - debiasfactor) ./ (imagsumW.^2 - debiasfactor)).';
end

% -------- baseline subtraction (full temporal resolution) --------
dwpli_bs = bsxfun(@minus, dwpli, mean(dwpli(:,baselineidxF(1):baselineidxF(2)), 2));

% -------- pack outputs --------
dwpli_struct = struct();
dwpli_struct.dwpli       = dwpli_bs;            % baseline-corrected
dwpli_struct.dwpli_raw   = dwpli;               % raw
dwpli_struct.freqs       = freqs2use(:)';
dwpli_struct.times       = times;
dwpli_struct.baseline_ms = baselinetm(:)';
dwpli_struct.channels    = {channel1, channel2};
dwpli_struct.params = struct( ...
    'min_cycles', min_cycles, ...
    'max_cycles', max_cycles, ...
    'num_cycles', num_cycles, ...
    'n_wavelet',  n_wavelet, ...
    'half_wavelet', half_wavelet, ...
    'n_convolution', n_convolution ...
);

end
