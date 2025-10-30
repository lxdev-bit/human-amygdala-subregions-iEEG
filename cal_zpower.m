function zpower_struct = cal_zpower(data, select_channels, freqs, fs, wavenum, baseline_offset, basline_win)
% Compute BOSC time-frequency power and z-baseline for a SINGLE dataset.
% No file I/O, no subject/condition info; concise and anonymized.
%
% Inputs:
%   data            : channels x time x trials
%   select_channels : channel indices to keep (e.g., [6 7])
%   freqs           : frequency vector (e.g., 1:180)
%   fs              : sampling rate in Hz (e.g., 1000)
%   wavenum         : BOSC wavelet parameter (e.g., 6)
%   baseline_offset : offset added to time axis in ms (e.g., 4000)
%   basline_win     : [t1 t2] ms baseline window relative to (time + offset), e.g., [-200 0]
%
% Output:
%   zpower_struct with fields:
%     .zpower      : trials x channels x freqs x time (z-scored power)
%     .freqs       : frequency vector
%     .T           : time vector in ms after offset
%     .selected_ch : channel indices used
%     .dims        : [n_trials, n_channels, n_freqs, n_time]

% --------- minimal defaults (if empty) ----------
if nargin < 2 || isempty(select_channels), select_channels = 1:size(data,1); end
if nargin < 3 || isempty(freqs),          freqs = 1:180; end
if nargin < 4 || isempty(fs),             fs = 1000;     end
if nargin < 5 || isempty(wavenum),        wavenum = 6;   end
if nargin < 6 || isempty(baseline_offset),baseline_offset = 4000; end
if nargin < 7 || isempty(basline_win),    basline_win = [-200 0];  end

% --------- select channels & sizes ----------
data = double(data(select_channels, :, :));        % ch x time x trials
[nchan, npts, ntrials] = size(data);

% --------- BOSC time-frequency power ----------
power_data = zeros(ntrials, nchan, numel(freqs), npts, 'double');
for i = 1:nchan
    for j = 1:ntrials
        tmp = data(i, :, j);                       % 1 x time
        [B, ~, ~] = BOSC_tf_power(tmp, freqs, fs, wavenum);
        power_data(j, i, :, :) = B;                % trials x ch x f x t
    end
end
clear data

% --------- baseline correction (zbaseline) ----------
t_ms = (0:(npts-1)) * (1000/fs);                   % ms
T = t_ms + baseline_offset;                        % align with offset
bt = find(T >= basline_win(1) & T < basline_win(2));
assert(~isempty(bt), 'Empty baseline window. Check baseline_offset and basline_win.');

wm.powspctrm   = power_data;
base.powspctrm = power_data(:,:,:,bt);
zpower_data    = zbaseline(wm, base);
clear wm base power_data

% --------- pack output ----------
zpower_struct = struct();
zpower_struct.zpower       = zpower_data;          % trials x ch x f x t
zpower_struct.freqs        = freqs;
zpower_struct.T            = T;                    % ms
zpower_struct.selected_ch  = select_channels;
zpower_struct.dims         = [ntrials, nchan, numel(freqs), npts];
end
