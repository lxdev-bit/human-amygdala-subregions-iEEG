function out = extra_point(z_cfg, dwpli_cfg, output_file)

% Extract fixed-window values for subsequent linear mixed-effects models.
%
% z-power and dwPLI:
%   Mean across predefined frequency bins and time samples.
% Each configuration is a structure array. Required fields are:
%
% z_cfg / dwpli_cfg:
%   .data         observation × frequency × time
%   .meta         metadata table with one row per observation
%   .freq_ind     selected frequency indices
%   .time_vec     time vector corresponding to data dimension 3
%   .windows      table with name, start_ms and end_ms
%   .region       region label
%   .condition    condition label
%   .contrast     optional contrast label

% OUTPUT
%   out.zpower
%   out.dwpli

out.zpower = extract_tf_metric(z_cfg, "zpower");
out.dwpli = extract_tf_metric(dwpli_cfg, "dwpli");

function dat = extract_tf_metric(cfg, metric_name)

dat = table();
if isempty(cfg), return; end

for i = 1:numel(cfg)
    D = cfg(i).data(:, cfg(i).freq_ind, :);
    n_obs = size(D, 1);

    for tw = 1:height(cfg(i).windows)
        tidx = cfg(i).time_vec >= cfg(i).windows.start_ms(tw) & cfg(i).time_vec <= cfg(i).windows.end_ms(tw);
        value = squeeze(mean(mean(D(:, :, tidx), 3, 'omitnan'), 2, 'omitnan'));

        tmp = cfg(i).meta;
        tmp.metric = repmat(string(metric_name), n_obs, 1);
        tmp.region = repmat(get_text(cfg(i), 'region'), n_obs, 1);
        tmp.condition = repmat(get_text(cfg(i), 'condition'), n_obs, 1);
        tmp.contrast = repmat(get_text(cfg(i), 'contrast'), n_obs, 1);
        tmp.timewin = repmat(string(cfg(i).windows.name(tw)), n_obs, 1);
        tmp.value = value;
        tmp.freq_start = repmat(cfg(i).freq_ind(1), n_obs, 1);
        tmp.freq_end = repmat(cfg(i).freq_ind(end), n_obs, 1);
        tmp.time_start = repmat(cfg(i).windows.start_ms(tw), n_obs, 1);
        tmp.time_end = repmat(cfg(i).windows.end_ms(tw), n_obs, 1);

        dat = [dat; tmp];
    end
end
end


function value = get_text(s, field_name)

if isfield(s, field_name) && ~isempty(s.(field_name))
    value = string(s.(field_name));
else
    value = "";
end
end
