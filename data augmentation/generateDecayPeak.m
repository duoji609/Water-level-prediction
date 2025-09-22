function charge=generateDecayPeak(peak_value)
    %% 参数设定
    % 持续时长范围（天）
    T_min = 8; T_max = 20;          % 降雨持续时间（天）
    % 随机生成持续时长（天）
    T = randi([T_min, T_max]);      % 持续时长（天）
    % 峰值大小范围（降雨强度）
    % peak_value_min = 500; peak_value_max = 2500; % 峰值大小范围（毫米/小时）
    % 随机生成峰值大小
    % peak_value = peak_value_min + (peak_value_max - peak_value_min) * rand; % 峰值大小
    % 时间参数
    t = 100;                  % 总时间步数
    dt = 1;                   % 时间步长
    time = 0:dt:t;            % 时间向量
    
    % 指数衰减参数
    Q0 = 100;                 % 初始流量
    lambda = 0.04+rand*0.05;            % 衰减系数
    
    % 单峰函数参数
    num_peaks = 10;           % 峰值数量
    peak_interval = 16+rand*5;        % 峰值之间的时间间隔
    peak_amplitude_initial = 50+40*rand; % 初始峰值幅度
    amplitude_decay =0.4+0.6*rand;    % 峰值幅度衰减因子
    peak_width = rand*3+0.5;           % 峰值宽度（标准差）
    
    %% 生成指数衰减曲线
    Q_base = Q0 * exp(-lambda * time);
    
    %% 生成多个单峰并叠加
    Q_peaks = zeros(size(time)); % 初始化峰值叠加向量
    
    for i = 1:num_peaks
        % 计算当前峰值的位置
        peak_time = (i-1) * peak_interval + 10; % 从时间点10开始添加峰值，避免一开始有峰
        if peak_time > max(time)
            break; % 如果峰值位置超出时间范围，停止添加
        end
        
        % 计算当前峰值的幅度
        current_amplitude = peak_amplitude_initial * (amplitude_decay)^(i-1);
        
        % 生成高斯峰值
        Q_peak = current_amplitude * exp(-((time - peak_time).^2) / (2 * peak_width^2));
        
        % 叠加到总峰值向量
        Q_peaks = Q_peaks + Q_peak;
    end
    
    %% 生成最终流量序列
    Q_total = Q_base + Q_peaks;
    
    % 调整分布以满足峰值大小
    Q = Q_total / max(Q_total) * peak_value; % 归一化后使峰值大小为指定范围内的随机值
    % figure;
    % plot(t, R, 'b-', 'LineWidth', 2);
    
    % % =============== 筛选大于 5% 峰值的部分 ===============
     threshold = 24; % 峰值的 5% 阈值
    % 
    % % 筛选出满足条件的时间点和对应降雨强度
     valid_indices = Q > threshold; % 找到满足条件的索引
     %t_filtered = t(valid_indices); % 筛选时间点
     Q_filtered = Q(valid_indices); % 筛选降雨强度
    % =============== 添加背景噪声 ===============
    noise_amplitude = 0.01 * peak_value; % 噪声幅度为峰值的 5%
    noise = noise_amplitude * randn(size(Q_filtered)); % 生成正态分布噪声
    
    % 将噪声添加到原始分布
    Q_noisy = Q_filtered + noise;
    
    % 确保噪声后的分布不出现负值（降雨强度不能为负）
    Q_noisy(Q_noisy < 24) = 24;
    
    % 使用添加噪声后的分布
    Q_filtered = Q_noisy;
    % figure;
    % plot(Q_filtered, 'b-', 'LineWidth', 2);
    
    num_samples = T;
    % 均匀采样的索引
    indices = round(linspace(1, length(Q_filtered), num_samples));
    % 采样结果
    charge = Q_filtered(indices);
    % figure;
    % bar(charge);
    % xlabel('天');
    % ylabel('流量值');
    % title('双峰流量过程图');
    % grid on;

    
    
    % %% 绘图展示
    % figure;
    % plot(time, Q_base, 'LineWidth', 2, 'DisplayName', '指数衰减基线');
    % hold on;
    % plot(time, Q_peaks, 'r--', 'LineWidth', 1.5, 'DisplayName', '叠加的峰值');
    % plot(time, Q_total, 'b-', 'LineWidth', 2, 'DisplayName', '最终流量序列');
    % xlabel('时间');
    % ylabel('流量');
    % title('具有多个逐渐减小峰值的流量序列');
    % legend('Location', 'northeast');
    % grid on;
end
