clc;
clear;
% MATLAB代码
% 从Excel文件中动态读取时间序列数据，并根据数据量自动处理存储到Excel

% 定义输入Excel文件和工作表信息
input_filename = '水位数据.xlsx'; % 输入的Excel文件名
sheet_name = 'Sheet4'; % 工作表名称

% 读取Excel中的数据
data = xlsread(input_filename, sheet_name); % 读取整个表格数据

% 获取数据的维度
[num_rows, num_series] = size(data); % num_rows 是数据的行数，num_series 是时间序列的数量

% 确保数据长度足够
if num_rows < 5
    error('每个时间序列的长度必须至少为5天');
end

% 计算输出的总行数
output_rows = num_rows - 5 + 1; % 每个时间序列的滑动窗口行数
output_columns = num_series * 5; % 每行包含5天数据，每个时间序列占5列

% 初始化结果矩阵
result_matrix = zeros(output_rows, output_columns);

% 按列（时间序列）循环处理数据
for series_idx = 1:num_series
    for i = 1:output_rows
        % 提取当前时间序列的第 i 到 i+3 天数据
        result_matrix(i, (series_idx-1)*5 + 1:series_idx*5) = data(i:i+4, series_idx);
    end
end

% 定义输出Excel文件名
output_filename = 'output_charge_data.xlsx';

% 将结果写入新的Excel文件
xlswrite(output_filename, result_matrix);

% 提示完成
disp(['数据已成功处理并写入 ', output_filename]);