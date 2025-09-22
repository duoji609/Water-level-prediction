function charge=generateDoublePeak(peak_value)
    % =============== 用户输入参数范围 ===============
    % 持续时长范围（天）
    T_min = 3; T_max = 15;          % 降雨持续时间（天）
    %peak_value = 1;
    % 不对称因子范围
    alpha_min = -5; alpha_max = 5;  % 不对称因子（负值左偏，正值右偏）
    
    % 分布均匀程度
    sigma_min = 0; sigma_max = 10; % 控制分布宽度，值越小越集中，越大越均匀
    
    % 峰值位置范围（天）
    % peak_min = 1; peak_max = 9;     % 峰值位置最小值和最大值（动态根据持续时长调整）
    
    % 峰值大小范围（降雨强度）
    % peak_value_min = 24; peak_value_max = 2000; % 峰值大小范围（毫米/小时）
    
    % =============== 随机生成参数 ===============
    % 随机生成持续时长（天）
    T = randi([T_min, T_max]);      % 持续时长（天）
    
    
    % 峰值位置范围动态调整为 [1, T] 的整数天
    peak_min = 2; % 峰值位置最小值，取整数
    peak_max = T-1;  % 峰值位置最大值，取整数
    mu = randi([peak_min, peak_max]);  % 随机生成整数天的峰值位置
    
    
    % 随机生成不对称因子
    alpha = alpha_min + (alpha_max - alpha_min) * rand; % 不对称因子
    
    % 随机生成标准差（控制分布宽度，影响均匀程度）
    sigma = sigma_min + (sigma_max - sigma_min) * rand; % 标准差
    
    t = linspace(mu-3*sigma, mu+3*sigma, 2000);          % 时间序列（天）
    
    
    % 随机生成峰值大小
    % peak_value = peak_value_min + (peak_value_max - peak_value_min) * rand; % 峰值大小
    
    % =============== 生成不对称正态分布 ===============
    % 不对称正态分布公式
    z = (t-mu) / sigma; % 标准化时间
    phi = exp(-z.^2 / 2) / sqrt(2*pi); % 标准正态分布 PDF
    Phi = 0.5 * (1 + erf(alpha * z / sqrt(2))); % 标准正态分布 CDF
    R = 2 * phi .* Phi; % 不对称正态分布 PDF
    
    % figure;
    % plot(t, phi, 'b-', 'LineWidth', 2);
    % figure;
    % plot(t, R, 'b-', 'LineWidth', 2);
    
    
    % 调整分布以满足峰值大小
    R = R / max(R) * peak_value; % 归一化后使峰值大小为指定范围内的随机值
    % figure;
    % plot(t, R, 'b-', 'LineWidth', 2);
    
    % % =============== 筛选大于 5% 峰值的部分 ===============
     threshold = 24; % 峰值的 5% 阈值
    % 
    % % 筛选出满足条件的时间点和对应降雨强度
     valid_indices = R > threshold; % 找到满足条件的索引
     t_filtered = t(valid_indices); % 筛选时间点
     R_filtered = R(valid_indices); % 筛选降雨强度
    
    % =============== 添加背景噪声 ===============
    noise_amplitude = 0.01 * peak_value; % 噪声幅度为峰值的 5%
    noise = noise_amplitude * randn(size(R_filtered)); % 生成正态分布噪声
    
    % 将噪声添加到原始分布
    R_noisy = R_filtered + noise;
    
    % 确保噪声后的分布不出现负值（降雨强度不能为负）
    R_noisy(R_noisy < 0) = 0;
    
    % 使用添加噪声后的分布
    R_filtered = R_noisy;
    % figure;
    % plot(t_filtered, R_filtered, 'b-', 'LineWidth', 2);
    
    
    
    [peakValue, idx_peak_current] = max(R_filtered);
    R_filtered_size = numel(R_filtered); 
    
    charge=zeros(1,T);
    
    center=floor((T+1)/2);
    charge(center)=peakValue;
    label1=idx_peak_current;
    label2=idx_peak_current;
    for i=1:(center-1)
        deltaT1=randi([1, floor((idx_peak_current-1)/(center-1))]);
        charge(center-i)=R_filtered(label1-deltaT1);
        label1=label1-deltaT1;
    end
    
    for i= 1:(T-center)
        deltaT2=randi([1, floor((R_filtered_size-idx_peak_current)/(T-center))]);
        charge(center+i)=R_filtered(label2+deltaT2);
        label2=label2+deltaT2;
    end
    
    % figure;
    % bar(charge);
    % xlabel('天');
    % ylabel('流量值');
    % title('单峰流量过程图');
    % grid on;
    % 
    % % % % =============== 输出结果 ===============
    % disp('生成的降雨分布参数：');
    % disp(['持续时长 (T): ', num2str(T), ' 天']);
    % % disp(['峰值位置 (mu): ', num2str(mu), ' 天']);
    % disp(['标准差 (sigma): ', num2str(sigma)]);
    % disp(['不对称因子 (alpha): ', num2str(alpha)]);
    % disp(['峰值大小: ', num2str(peak_value), ' 毫米/小时']);
end