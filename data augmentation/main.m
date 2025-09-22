clc;
clear;
close all;

charge1=generateDecayPeak(500+rand*2000/3);
charge2=generateDecayPeak(500+2000/3+rand*2000/3);
charge3=generateDecayPeak(500+4000/3+rand*2000/3);

charge4=generateSinglePeak(24+rand*1900/3);
charge5=generateSinglePeak(24+1900/3+rand*1900/3);
charge6=generateSinglePeak(24+1900/3*2+rand*1900/3);

charge7=generateDoublePeak(500+rand*2000/3);
charge8=generateDoublePeak(500+2000/3+rand*2000/3);
charge9=generateDoublePeak(500+4000/3+rand*2000/3);

% 将数组存储在 cell 数组中
charge = {[0,charge1,0], [0,charge2,0],[0,charge3,0],......
          [0,charge4,0],[0,charge5,0],[0,charge6,0],......
          [0,charge7,0],[0,charge8,0],[0,charge9,0]};

% 随机生成拼接顺序
num_charge = numel(charge);    % 数组数量
random_order = randperm(num_charge); % 生成随机顺序

% 初始化结果数组
result = [];

% 按随机顺序拼接数组
for i = 1:num_charge
    result = [result, charge{random_order(i)}]; % 水平拼接
end

figure;
bar(result);
xlabel('天');
ylabel('流量值');
title('流量过程图');
grid on;

% 将数组写入 Excel 文件
filename = 'generateCharge3.xlsx'; % Excel 文件名
writematrix(result', filename);

disp(['数组已保存到文件：', filename]);