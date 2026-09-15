%{
Script: EEG_ERSP_analysis.m

Purpose
-------
Preprocess FineMI EEG recordings and characterize task-related
time-frequency activity during unilateral multi-joint motor imagery
using event-related spectral perturbation (ERSP) analysis.

The analysis includes:
1. Loading and resampling raw EEG recordings from 18 participants.
2. Band-pass filtering the EEG signals from 4 to 40 Hz.
3. Applying common average reference (CAR).
4. Correcting dataset-specific event annotations and segmenting trials.
5. Computing ERSPs for a predefined nine-channel left sensorimotor ROI
   using time-frequency decomposition with a fixed cycle number of 5.
6. Applying baseline normalization using the pre-MI interval from
   -4 to -2 s.
7. Averaging ERSP power across the nine ROI channels in the
   time-frequency domain.
8. Averaging ERSPs across participants for each of the eight
   motor-imagery conditions.
9. Generating and exporting task-specific time-frequency maps.

Usage
-----
1. Put the FineMI EEG dataset in the `Data/FineMI/` folder. 
2. Update the data and output paths below if necessary.
3. Run this script from beginning to end.

Input
-----
Raw FineMI EEG recordings (.cnt).

Output
------
- Preprocessed participant-level EEG data (.mat).
- Participant- and condition-level ERSP results (.mat).
- Grand-averaged ERSP time-frequency figures for the eight
  motor-imagery conditions (.png).

All output files are saved in the `matlab_files1/` folder.

Important
---------
The ERSP analysis uses a predefined nine-channel left sensorimotor ROI:
FC5, FC3, FC1, C5, C3, C1, CP5, CP3, and CP1.

ERSPs are computed independently for each channel and then averaged
in the time-frequency domain to avoid phase cancellation associated
with averaging signals in the time domain.

Do not modify the sampling rate, filtering parameters, epoch definition,
baseline interval, wavelet-cycle setting, or ROI channel configuration
unless the analysis plan is intentionally revised.

Author:AN Long
Date:2026-09-15
%}
%% 初始化与文件夹创建
clear; clc;
folder = "matlab_files1";
if ~exist(folder, 'dir'), mkdir(folder); end

fs = 250;
tmin = -4; % epoch 开始时间(s)
t_window_length = 10; % trial 长度(s)
win_start = tmin * fs; 
win_len = t_window_length * fs - 1;
lpass = 4; hpass = 40;
[f_b, f_a] = butter(3, [2*lpass/fs, 2*hpass/fs]); % 3阶带通滤波 4-40Hz

%% Subject 1 数据加载、预处理与分段
disp('Processing Subject 1...');
blocks = {'block1-4', 'block5', 'block6', 'block7', 'block8'};
data_all = []; label_all = [];

for b = 1:length(blocks)
    EEG = pop_loadcnt(['./Data/FineMI/subject1/EEG/', blocks{b}, '.cnt'], 'dataformat', 'int32');
    EEG = pop_resample(eeg_checkset(EEG), fs);
    data = double(EEG.data);
    data([33 43 65 66 67 68], :) = []; % 剔除无效通道
    
    % 滤波与 CAR(公共平均参考)
    for j = 1:62, data(j,:) = filtfilt(f_b, f_a, data(j,:)); end
    data = data - mean(data, 1); 
    
    % Epoching 分段
    n_trials = length(EEG.event);
    data_epoched = zeros(62, win_len + 1, n_trials);
    labels = zeros(n_trials, 1);
    for i = 1:n_trials
        lat = EEG.event(1,i).latency;
        labels(i) = EEG.event(1,i).type;
        data_epoched(:,:,i) = data(:, lat+win_start : lat+win_start+win_len);
    end
    
    data_all = cat(3, data_all, data_epoched);
    label_all = [label_all; labels];
end
data = reshape(data_all, 62, []); % 还原为二维方便后续处理
label = label_all;
save('matlab_files1/subject1.mat', 'data', 'label');

%% Subject 2-18 数据加载、预处理与分段
for sub = 2:18
    disp(['Processing Subject ', num2str(sub), '...']);
    data_all = []; label_all = [];
    
    for b = 1:8
        EEG = pop_loadcnt(sprintf('./Data/FineMI/subject%d/EEG/block%d.cnt', sub, b), 'dataformat', 'int32');
        EEG = pop_resample(eeg_checkset(EEG), fs);
        data = double(EEG.data);
        data([33 43 65 66 67 68], :) = [];
        
        for j = 1:62, data(j,:) = filtfilt(f_b, f_a, data(j,:)); end
        data = data - mean(data, 1);
        
        % 修正Subject 5 Block 6 第一个trial异常的问题
        evt_start = (b == 6 && sub == 5) * 2 + (b ~= 6 || sub ~= 5) * 1; 
        n_trials = length(evt_start:length(EEG.event));
        
        data_epoched = zeros(62, win_len + 1, n_trials);
        labels = zeros(n_trials, 1);
        idx = 1;
        for i = evt_start:length(EEG.event)
            lat = EEG.event(1,i).latency;
            labels(idx) = EEG.event(1,i).type;
            data_epoched(:,:,idx) = data(:, lat+win_start : lat+win_start+win_len);
            idx = idx + 1;
        end
        data_all = cat(3, data_all, data_epoched);
        label_all = [label_all; labels];
    end
    data = reshape(data_all, 62, []);
    label = label_all;
    save(sprintf('matlab_files1/subject%d.mat', sub), 'data', 'label');
end

%% 计算 ERSP (9通道在时频域求平均 - 修正相位抵消问题)
disp('Computing ERSP for 9-Channel Average (TF-domain averaging)...');
n_classes = 8;
frames = fs * t_window_length;
tlimits = [-4000, 6000]; % 时间范围
cycles = 5; 

% 对应通道: FC5(14), FC3(15), FC1(16), C5(24), C3(26), C1(27), CP5(34), CP3(35), CP1(36)
ch_indices = [14, 15, 16, 24, 26, 27, 34, 35, 36]; 
n_chans = length(ch_indices);

ERSP_all_subjects = cell(18, n_classes);

for sub = 1:18
    load(sprintf('matlab_files1/subject%d.mat', sub), 'data', 'label');
    for c = 1:n_classes 
        idx = find(label == c);
        trials = length(idx);
        
        % 初始化该被试、该类别所有通道的累加 ERSP 矩阵
        ERSP_avg_chans = 0; 
        
        % 遍历 9 个感兴趣的通道，独立计算每个通道的 ERSP
        for ch_idx = 1:n_chans
            curr_ch = ch_indices(ch_idx);
            temp_data = zeros(1, frames * trials);
            
            % 提取单通道数据并拼接 trials
            for k = 1:trials
                temp_data(1, (1:frames) + frames*(k-1)) = data(curr_ch, (1:frames) + frames*(idx(k)-1));
            end
            
            % 核心修正：单通道独立计算 ERSP (明确基线为 -4000 到 -2000 ms)
            [ERSP_single_ch, ~, ~, times, freqs] = q_timef(temp_data, frames, tlimits, fs, cycles, 'baseline', -2000); 
            
            % 将计算好的单通道频域能量累加
            ERSP_avg_chans = ERSP_avg_chans + ERSP_single_ch;
        end
        
        % 对 9 个通道的频域能量求平均 (避免时域相加导致的相位抵消)
        ERSP_avg_chans = ERSP_avg_chans / n_chans;
        
        % 存入最终矩阵
        ERSP_all_subjects{sub, c} = ERSP_avg_chans;
    end
end
% 保存固定 cycle=5 的结果
save(sprintf('matlab_files1/ersp_9chan_avg_cyc%d_all.mat', cycles), 'ERSP_all_subjects', 'times', 'freqs');


%% 绘制平均 ERSP 时频图 (导出8张图)
disp('Plotting Averaged ERSP for 9-Channel Average...');

class_names = ["Hand open/close", "Wrist flexion/extension", "Wrist abduction/adduction", ...
               "Elbow pronation/supination", "Elbow flexion/extension", ...
               "Shoulder pronation/supination", "Shoulder abduction/adduction", ...
               "Shoulder flexion/extension"];
file_class_names = strrep(class_names, '/', '_');
file_class_names = strrep(file_class_names, ' ', '_');

figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.6, 0.6]);

cycles = 5;
load(sprintf('matlab_files1/ersp_9chan_avg_cyc%d_all.mat', cycles));

% 动态获取 35Hz 以下的频率索引
freq_idx = find(freqs <= 35); 

for c = 1:n_classes
    % 提取并对18个受试者求平均
    ERSP_sum = zeros(size(ERSP_all_subjects{1, c}));
    for sub = 1:18
        ERSP_sum = ERSP_sum + ERSP_all_subjects{sub, c};
    end
    ERSP_mean = ERSP_sum / 18;
    
    clf; % 清空当前图窗
    pcolor(times/1000, freqs(freq_idx), ERSP_mean(freq_idx, :)); 
    shading interp;
    
    % 设置轴域及标线
    xticks(-4:2:6);
    xticklabels(["-4","-2", "0", "2", "4","6"]);
    hold on;
    
    % 绘制时间分割虚线 (动作提示 -2s / 开始 0s / 结束 4s)
    plot([-2 -2], [freqs(1) freqs(freq_idx(end))], 'k--', 'LineWidth', 2);
    plot([0 0], [freqs(1) freqs(freq_idx(end))], 'k--', 'LineWidth', 2);
    plot([4 4], [freqs(1) freqs(freq_idx(end))], 'k--', 'LineWidth', 2);
    
    caxis([-3 1]); % 颜色标尺
    colorbar('vert', 'fontsize', 22, 'fontweight', 'b');
    
    xlabel('Time (s)', 'fontsize', 30, 'fontweight', 'b');
    ylabel('Frequency (Hz)', 'fontsize', 30, 'fontweight', 'b');
    
    % 动态修改标题
    title(sprintf('%s',class_names(c)), 'fontsize', 38);
    set(gca, 'fontsize', 24, 'fontweight', 'b');
    
    % 保存图片
    pic_name = sprintf("matlab_files1/TF_18subj_C3_avg_cyc%d_Class%d_%s.png", cycles, c, file_class_names(c));
    exportgraphics(gcf, pic_name, "Resolution", 600);
end

disp('All tasks completed successfully!');