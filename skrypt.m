clear; close all; clc;

% Nagranie dwóch osób mówiących naprzemiennie wieczorem - Jakub i Edyta (studentka) 
[a, Fs] = audioread('wieczor.mp3');

prop = audioFeatureExtractor('SampleRate', Fs, ...
    'SpectralDescriptorInput','barkSpectrum', ...
    'pitch', true, ...
    'spectralFlux', true, ...
    'harmonicRatio', true, ...
    'spectralSpread', true);

wyn = extract(prop, a);

pitch = wyn(:,3); 

% Uśrednianie pitch w oknach
window_size = 95;
n_windows = floor(length(pitch) / window_size);

avg_pitch = zeros(n_windows,1);

for i = 1:n_windows
    idx_start = (i-1) * window_size + 1;
    idx_end = i * window_size;
    window = pitch(idx_start:idx_end);
    window = window(window > 0 & ~isnan(window)); 

    if isempty(window)
        avg_pitch(i) = NaN;
    else
        avg_pitch(i) = mean(window);
    end
end

% Wygładzenie filtrem medianowym
smoothed_pitch = medfilt1(avg_pitch, 3, 'omitnan', 'truncate');

t_pitch = (0:length(smoothed_pitch)-1) * (window_size / Fs);
figure;
plot(t_pitch, smoothed_pitch);
xlabel('Czas');
ylabel('Pitch [Hz]');
title('Pitch w czasie');
grid on;

% Rozciągnięcie
expanded_pitch = repelem(smoothed_pitch, window_size);

if length(expanded_pitch) < length(pitch)
    expanded_pitch(length(pitch)) = NaN;
elseif length(expanded_pitch) > length(pitch)
    expanded_pitch = expanded_pitch(1:length(pitch));
end

t = (0:length(pitch)-1) / Fs;
figure;
plot(t, pitch, 'b'); hold on;
plot(t, expanded_pitch, 'r', 'LineWidth', 2);
xlabel('Czas [s]');
ylabel('Pitch [Hz]');
title('Pitch oryginalny vs expanded pitch');
legend('Oryginalny pitch', 'Expanded pitch');
grid on;

% Segmentacja Jakuba 
% (docelowo wyciągane są fragmenty Edyty ale w ten
% sposób wyniki były lepsze) 
jakub = nan(size(expanded_pitch));
jakub(expanded_pitch < 190) = 1;

hop = 512;
segments = [];
inSpeech = false;

for i = 1:length(jakub)
    if jakub(i) == 1 && ~inSpeech
        start = (i - 1) * hop + 1;
        inSpeech = true;
    elseif jakub(i) ~= 1 && inSpeech
        finish = (i - 1) * hop + 1;
        segments = [segments; start, finish];
        inSpeech = false;
    end
end

if inSpeech
    finish = length(a);
    segments = [segments; start, finish];
end

% Filtr długości (10s)
min_dur = Fs * 10.0; 
segments = segments((segments(:,2) - segments(:,1)) >= min_dur, :);

% Maski
mask = zeros(size(a));
for i = 1:size(segments,1)
    mask(segments(i,1):min(segments(i,2), length(a))) = 1;
end

jakub_audio = a .* mask;

% Docelowe segmenty
edyta_mask = ~mask;
edyta_audio = a .* edyta_mask;

% Wykres sygnału audio, podzielony na rozmówców
t = (0:length(a)-1)/Fs;
figure;
plot(t, edyta_audio, 'r', t, jakub_audio, 'b'); 
xlabel('Czas [s]'); 
ylabel('Amplituda'); 
title('Sygnał audio (Niebieski - Jakub, Czerwony - Edyta)'); 
grid on;

fprintf('\nMoje segmenty (Edyta)\n');

current_pos = 1;

for i = 1:size(segments, 1)
    start_edyta = segments(i, 1);

    % Jeśli między obecną pozycją a startem Jakuba jest przerwa > 0.5s, to
    % uznajemy, że to Edyta - znajomość nagrania na to pozwala
    if (start_edyta - current_pos) > (Fs * 0.5)
        s_time = current_pos / Fs;
        e_time = (start_edyta - 1) / Fs;
        fprintf('Segment: %.2f s - %.2f s (czas trwania: %.2f s)\n', ...
            s_time, e_time, e_time - s_time);
    end

    % Przesunięcie kursora na koniec obecnego segmentu Jakuba
    current_pos = segments(i, 2) + 1;
end


%% ILOŚĆ SŁÓW

segments_edyta = [];
inSpeechEdyta = false;
startE = 1;

for i = 1:length(edyta_mask)
    if edyta_mask(i) == 1 && ~inSpeechEdyta
        startE = i;
        inSpeechEdyta = true;
    elseif edyta_mask(i) == 0 && inSpeechEdyta
        finishE = i - 1;
        segments_edyta = [segments_edyta; startE, finishE];
        inSpeechEdyta = false;
    end
end

if inSpeechEdyta
    segments_edyta = [segments_edyta; startE, length(edyta_mask)];
end

min_dur_e = Fs * 2.0; 
segments_edyta = segments_edyta((segments_edyta(:,2) - segments_edyta(:,1)) >= min_dur_e, :);

window_len = round(Fs * 0.03); 
energy_e = envelope(edyta_audio, window_len, 'rms');

zcr_e = zeros(size(edyta_audio));
step_zcr = round(Fs * 0.01); 

for i = 1:step_zcr:length(edyta_audio)-window_len
    win = edyta_audio(i:i+window_len);
    zcr_e(i:i+step_zcr) = sum(abs(diff(sign(win)))) / (2 * window_len);
end

words_real_edyta = [45, 56, 47, 42]; 

% Parametry dla każdego segmentu
thresh_vals = [0.06, 0.05, 0.20, 0.35];
dist_vals   = [0.19, 0.25, 0.15, 0.19];

best_params_per_segment = struct('thresh', {}, 'dist', {}, 'error', {});
best_counts = zeros(size(segments_edyta,1),1);

word_positions = cell(size(segments_edyta,1),1);
time_vectors = cell(size(segments_edyta,1),1);
features_all = cell(size(segments_edyta,1),1);

for i = 1:size(segments_edyta,1)

    s_idx = segments_edyta(i,1);
    f_idx = min(segments_edyta(i,2), length(energy_e));

    combined_feature = energy_e(s_idx:f_idx) + (zcr_e(s_idx:f_idx) * 0.1);
    feat_smooth = smoothdata(combined_feature, 'gaussian', round(Fs*0.06));

    best_th = thresh_vals(i);
    best_dist = dist_vals(i);

    l_thresh = mean(feat_smooth) + best_th * std(feat_smooth);

    [~, locs] = findpeaks(feat_smooth, ...
        'MinPeakHeight', l_thresh, ...
        'MinPeakDistance', round(Fs * best_dist));

    t_local = (s_idx:f_idx)/Fs;

    word_positions{i} = locs;
    time_vectors{i} = t_local;
    features_all{i} = feat_smooth;

    best_count = length(locs);
    real_val = words_real_edyta(i);
    best_error_local = abs(real_val - best_count) / real_val * 100;

    best_params_per_segment(i).thresh = best_th;
    best_params_per_segment(i).dist = best_dist;
    best_params_per_segment(i).error = best_error_local;
    best_counts(i) = best_count;
end

fprintf('\nWyniki liczenia słów dla każdego segmentu\n');

bledy_fragmentow = [best_params_per_segment.error]; 

for i = 1:length(best_counts)
    real_val = words_real_edyta(i);
    err = bledy_fragmentow(i);

    fprintf(['Segment %d:\n' ...
        '  thresh = %.2f\n' ...
        '  dist   = %.2f s\n' ...
        '  wykryto = %d | realnie = %d\n' ...
        '  błąd fragmentu = %.2f%%\n\n'], ...
        i, ...
        best_params_per_segment(i).thresh, ...
        best_params_per_segment(i).dist, ...
        best_counts(i), real_val, err);
end

licznik = sum(bledy_fragmentow .* words_real_edyta);
mianownik = sum(words_real_edyta);
sredni_blad = licznik / mianownik;

fprintf('Średni błąd: %.2f%%\n', sredni_blad);

figure;

for i = 1:size(segments_edyta,1)

    subplot(4,1,i);

    t_local = time_vectors{i};
    feat = features_all{i};
    locs = word_positions{i};

    plot(t_local, feat, 'b'); hold on;

    for k = 1:length(locs)
        x = t_local(locs(k));
        xline(x, 'r');
    end

    xlabel('Czas [s]');
    ylabel('Energia + ZCR');
    title(['Segment ', num2str(i), ' - detekcja słów']);
    grid on;
end
 
% Wyniki liczenia słów dla każdego segmentu
% Segment 1:
%   thresh = 0.06
%   dist   = 0.19 s
%   wykryto = 44 | realnie = 45
%   błąd fragmentu = 2.22%
% 
% Segment 2:
%   thresh = 0.05
%   dist   = 0.25 s
%   wykryto = 61 | realnie = 56
%   błąd fragmentu = 8.93%
% 
% Segment 3:
%   thresh = 0.20
%   dist   = 0.15 s
%   wykryto = 48 | realnie = 47
%   błąd fragmentu = 2.13%
% 
% Segment 4:
%   thresh = 0.35
%   dist   = 0.19 s
%   wykryto = 42 | realnie = 42
%   błąd fragmentu = 0.00%
% 
% Średni błąd: 3.68%

%% Zadanie 2 

clear; close all; clc;

[b, Fs_1] = audioread('wieczor_ostatni_fragment.mp3');
[c, Fs_2] = audioread('rano_fragment.mp3');

signals = {b, c};
Fs_all = [Fs_1, Fs_2];
names = {'wieczor', 'rano'};

for s = 1:2

    signal = signals{s};
    Fs = Fs_all(s);

    window_len = round(Fs * 0.03); 
    energy = envelope(signal, window_len, 'rms');

    zcr = zeros(size(signal));
    step_zcr = round(Fs * 0.01); 

    for i = 1:step_zcr:length(signal)-window_len
        win = signal(i:i+window_len);
        zcr(i:i+step_zcr) = sum(abs(diff(sign(win)))) / (2 * window_len);
    end

    combined = energy + 0.1 * zcr;
    feat_smooth = smoothdata(combined, 'gaussian', round(Fs*0.06));
 
    if s == 1   
        th = 0.35;
        dist = 0.190;
    else       
        th = 0.10;
        dist = 0.210;
    end

    l_thresh = mean(feat_smooth) + th * std(feat_smooth);

    [~, locs, widths] = findpeaks(feat_smooth, ...
        'MinPeakHeight', l_thresh, ...
        'MinPeakDistance', round(Fs * dist));

    best_locs = locs;
    best_widths = widths;

    count = length(locs);
    real_words = 42;
    best_error = abs(real_words - count) / real_words * 100;

    best_th = th;
    best_dist = dist;
    
    fprintf('\nFragment: %s \n', names{s});

    word_segments = [];


    for i = 1:length(best_locs)

        center = best_locs(i);
        width = best_widths(i);

        start_idx = max(1, round(center - width/2));
        end_idx   = min(length(signal), round(center + width/2));

        word_segments = [word_segments; start_idx, end_idx];

        fprintf('Słowo %d: %.2f s - %.2f s\n', ...
            i, start_idx/Fs, end_idx/Fs);
    end

    fprintf('Liczba wykrytych słów: %d\n', length(best_locs));
    fprintf('Błąd: %.2f%%\n', best_error);
    fprintf('\nParametry:\n');
    fprintf('thresh: %.2f\n', best_th);
    fprintf('dist: %.3f s\n', best_dist);
    

    durations = (word_segments(:,2) - word_segments(:,1)) / Fs;

    fprintf('\nCzas trwania słów:\n');
    fprintf('Min: %.3f s\n', min(durations));
    fprintf('Max: %.3f s\n', max(durations));
    fprintf('Średnia: %.3f s\n', mean(durations));

    faster = durations < mean(durations);
    slower = durations > mean(durations);

    fprintf('Szybsze niż średnia: %d\n', sum(faster));
    fprintf('Wolniejsze niż średnia: %d\n', sum(slower));

    pauses = [];

    for i = 1:size(word_segments,1)-1
        pause = (word_segments(i+1,1) - word_segments(i,2)) / Fs;
        pauses = [pauses; pause];
    end

    fprintf('\nPrzerwy:\n');
    fprintf('Min: %.3f s\n', min(pauses));
    fprintf('Max: %.3f s\n', max(pauses));
    fprintf('Średnia: %.3f s\n', mean(pauses));

    trend = polyfit(1:length(pauses), pauses', 1);

    if trend(1) > 0
        fprintf('Przerwy rosną (mówca zwalnia)\n');
    else
        fprintf('Przerwy maleją (mówca przyspiesza)\n');
    end

    prop = audioFeatureExtractor('SampleRate', Fs, ...
    'pitch', true);

    features = extract(prop, signal);
    pitch = features(:,1);

    pitch = repelem(pitch, ceil(length(signal)/length(pitch)));
    pitch = pitch(1:length(signal));

    fprintf('\nPitch (pierwsze 10 słów)\n');

    for i = 1:min(10, size(word_segments,1))

        s_idx = word_segments(i,1);
        e_idx = word_segments(i,2);

        pitch_seg = pitch(s_idx:e_idx);
        pitch_seg = pitch_seg(pitch_seg > 0 & ~isnan(pitch_seg));

        if isempty(pitch_seg)
            fprintf('Słowo %d: brak danych pitch\n', i);
            continue;
        end

        p_min = min(pitch_seg);
        p_max = max(pitch_seg);
        p_mean = mean(pitch_seg);
        p_range = p_max - p_min;

        fprintf(['Słowo %d:\n' ...
            '  min = %.1f Hz\n' ...
            '  max = %.1f Hz\n' ...
            '  średni = %.1f Hz\n'...
            '  zmienność tonu = %.1f Hz\n'], ...
            i, p_min, p_max, p_mean, p_range);

    end
end


