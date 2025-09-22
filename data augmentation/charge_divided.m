% MATLAB 程序
% 功能：从 Excel 文件中读取若干列的时间序列数据，
% 由用户指定时间间隔和等分子序列数，插值数据并计算每段的水体总量。

% 清空工作区
clc; clear;

% === 用户输入参数 ===
excelFileName = input('请输入Excel文件名（包括扩展名）：', 's');
sheetName = input('请输入工作表名称：', 's');
columnRange = input('请输入读取列范围（如 A:C）：', 's');
timeInterval = input('请输入两个数据点之间的时间间隔（单位：小时）：');
numSubIntervals = input('请输入划分的时段数（例如：10）：');

% === 读取Excel文件 ===
data = readmatrix(excelFileName, 'Sheet', sheetName, 'Range', columnRange);

% 检查数据是否为空
if isempty(data)
    error('从Excel读取的数据为空，请检查文件和列范围！');
end

% 获取列数
[~, numCols] = size(data);

% 初始化结果存储
segmentVolume = {};

% === 插值处理与计算 ===
for col = 1:numCols
    % 提取当前列数据，去除 NaN
    currentData = data(:, col);
    currentData = currentData(~isnan(currentData));

    % 如果当前列无有效数据，跳过
    if isempty(currentData)
        continue;
    end

    % 原始时间轴
    numRows = length(currentData);
    originalTime = (0:numRows-1) * timeInterval;

    % 新的时间轴与插值
    totalDuration = originalTime(end); % 总持续时间
    newTimeInterval = totalDuration / numSubIntervals; % 每段时间的长度
    newTime = linspace(0, totalDuration, numSubIntervals + 1); % 等分时间轴
    interpolatedData = interp1(originalTime, currentData, newTime, 'linear');

    % 计算每段的水体总量
    segmentVolumes = zeros(numSubIntervals, 1);
    for segment = 1:numSubIntervals
        % 每段的起止时间
        startTime = newTime(segment);
        endTime = newTime(segment + 1);

        % 当前段的流量强度（线性插值结果）
        avgFlow = (interpolatedData(segment) + interpolatedData(segment + 1)) / 2; % 平均流量
        segmentVolumes(segment) = avgFlow * newTimeInterval; % 水体总量 = 平均流量 * 时间间隔
    end

    % 存储结果
    segmentVolume{col} = segmentVolumes;
end

% === 保存结果 ===
outputFileName = 'segment_volume_output.xlsx';
for col = 1:length(segmentVolume)
    if ~isempty(segmentVolume{col})
        writematrix(segmentVolume{col}, outputFileName, 'Sheet', ['Sequence_' num2str(col)]);
    end
end

disp(['计算结果已保存到文件：', outputFileName]);