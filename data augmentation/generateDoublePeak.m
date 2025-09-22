function charge=generateDoublePeak(peak_value)
    % =============== 用户输入参数范围 ===============
    % 持续时长范围（天）
    T_min = 24; T_max = 60;          % 降雨持续时间（天）
    % 随机生成持续时长（天）
    T = randi([T_min, T_max]);      % 持续时长（天）
    % 不对称因子范围

    alpha_min = -8; alpha_max = 8;  % 不对称因子（负值左偏，正值右偏）
    
    % 分布均匀程度
    sigma_min = 1; sigma_max = 2; % 控制分布宽度，值越小越集中，越大越均匀
    % 峰值大小范围（降雨强度）
    % peak_value_min = 500; peak_value_max = 2500; % 峰值大小范围（毫米/小时）
    % 随机生成峰值大小
    % peak_value = peak_value_min + (peak_value_max - peak_value_min) * rand; % 峰值大小
    % =============== 随机生成参数 ===============
    % 随机生成不对称因子
    alpha1 = alpha_min + (alpha_max - alpha_min) * rand; % 不对称因子
    alpha2 = alpha_min + (alpha_max - alpha_min) * rand; % 不对称因子
    % 随机生成标准差（控制分布宽度，影响均匀程度）
    sigma1 = sigma_min + (sigma_max - sigma_min) * rand; % 标准差
    sigma2 = sigma_min + (sigma_max - sigma_min) * rand; % 标准差
    mu1=0;
    mu2= 1.5*(sigma1+sigma2) + (0.1*(sigma1+sigma2))*rand;
    % 随机生成标准差（控制分布宽度，影响均匀程度）
    t = linspace(-3*sigma1, mu2+3*sigma1, 4000);          % 时间序列（天）

    % =============== 生成不对称正态分布 ===============
    % 峰1
    z1 = (t-mu1) / sigma1; % 标准化时间
    phi1 = exp(-z1.^2 / 2) / sqrt(2*pi); % 标准正态分布 PDF
    Phi1 = 0.5 * (1 + erf(alpha1 * z1 / sqrt(2))); % 标准正态分布 CDF
    R1 = (0.4+0.6*rand)* 2 * phi1 .* Phi1; % 不对称正态分布 PDF

    % 峰2
    z2 = (t-mu2) / sigma2; % 标准化时间
    phi2 = exp(-z2.^2 / 2) / sqrt(2*pi); % 标准正态分布 PDF
    Phi2 = 0.5 * (1 + erf(alpha2 * z2 / sqrt(2))); % 标准正态分布 CDF
    R2 = (0.4+0.6*rand)* 2 * phi2 .* Phi2; % 不对称正态分布 PDF
    
    R=R1+R2;

    % figure;
    % plot(t, R, '-b', 'LineWidth', 1.5); hold on;
    % plot(t, R1, '--r', 'LineWidth', 1); % 峰1
    % plot(t, R2, '--g', 'LineWidth', 1); % 峰2
    % hold off;


    % 调整分布以满足峰值大小
    R = R / max(R) * peak_value; % 归一化后使峰值大小为指定范围内的随机值
    % figure;
    % plot(t, R, 'b-', 'LineWidth', 2);
    
    % % =============== 筛选大于 5% 峰值的部分 ===============
     threshold = 24; % 峰值的 5% 阈值
    % 
    % % 筛选出满足条件的时间点和对应降雨强度
     valid_indices = R > threshold; % 找到满足条件的索引
     %t_filtered = t(valid_indices); % 筛选时间点
     R_filtered = R(valid_indices); % 筛选降雨强度
    % =============== 添加背景噪声 ===============
    noise_amplitude = 0.05 * peak_value; % 噪声幅度为峰值的 5%
    noise = noise_amplitude * randn(size(R_filtered)); % 生成正态分布噪声
    
    % 将噪声添加到原始分布
    R_noisy = R_filtered + noise;
    
    % 确保噪声后的分布不出现负值（降雨强度不能为负）
    R_noisy(R_noisy < 0) = 0;
    
    % 使用添加噪声后的分布
    R_filtered = R_noisy;
    % figure;
    % plot(t_filtered, R_filtered, 'b-', 'LineWidth', 2);
    
    num_samples = T;
    % 均匀采样的索引
    indices = round(linspace(1, length(R_filtered), num_samples));
    % 采样结果
    charge = R_filtered(indices);
    figure;
    bar(charge);
    xlabel('天');
    ylabel('流量值');
    title('双峰流量过程图');
    grid on;
end