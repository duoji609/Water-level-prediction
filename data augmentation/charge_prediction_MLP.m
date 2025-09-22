clc;
clear;

% 用户输入用于生成输入样本的历史天数
% numDays = input('请输入历史天数 (至少为 1)： ');
numDays = 1;

if numDays < 1
    error('历史天数必须大于等于 1！');
end

% 定义保存的网络文件名
networkFileName = 'MLP_trained_model.mat';
falseFileName = 'false_trained_model.mat';

% 判断是否存在已保存的网络
% if isfile(networkFileName)
if isfile(networkFileName)
    % 加载训练好的网络和标准化参数
    load('MLP_trained_model.mat', 'net');
    load('MLP_NormalizationParameters.mat', 'X_mean', 'X_std', 'Y_mean', 'Y_std');

    % 加载流量数据和初始水位值
    Q = readmatrix("generateCharge.xlsx", 'Sheet', "Sheet1", 'Range', 'A:A');
    H_init = 31.40;
    numSteps = length(Q) - numDays;
    predicted_H = zeros(numSteps, 1);
    current_H = H_init;

    for t = 1:numSteps
        if t == 1
            input_sample = [Q(1:numDays)', Q(numDays+1), current_H];
        else
            input_sample = [Q(t:t+numDays-1)', Q(t+numDays), current_H];
        end

        input_sample = (input_sample - X_mean) ./ X_std;
        predicted_H_norm = predict(net, input_sample);
        predicted_H(t) = predicted_H_norm * Y_std + Y_mean;
        current_H = predicted_H(t);
    end

    disp('递归预测的水位值：');
    disp(predicted_H);

    writematrix(predicted_H, "waterLevelPrediction.xlsx");

    figure;
    plot(1:numSteps, predicted_H, 'r-o', 'LineWidth', 1.5);
    title('递归预测水位值');
    xlabel('预测天数');
    ylabel('水位值');
    grid on;

else
    disp('未找到保存的网络，开始训练新网络...');

    data = readmatrix("流量水位预测.xlsx", 'Sheet', "Sheet1");
    [numRows, numCols] = size(data);
    numStations = numCols / 2;

    X_all = [];
    Y_all = [];

    for station = 1:numStations
        Q = data(:, (station-1)*2 + 1);
        H = data(:, (station-1)*2 + 2);
        validIdx = ~isnan(Q) & ~isnan(H);
        Q = Q(validIdx);
        H = H(validIdx);

        numSamples = length(Q) - numDays;
        numFeatures = numDays * 2 + 1;
        X_station = zeros(numSamples, numFeatures);
        Y_station = zeros(numSamples, 1);

        for t = numDays+1:length(Q)
            X_station(t-numDays, :) = [Q(t-numDays:t-1)', Q(t), H(t-numDays:t-1)'];
            Y_station(t-numDays) = H(t);
        end

        X_all = [X_all; X_station];
        Y_all = [Y_all; Y_station];
    end

    X_mean = mean(X_all, 1);
    X_std = std(X_all, 0, 1);
    X_all = (X_all - X_mean) ./ X_std;

    Y_mean = mean(Y_all);
    Y_std = std(Y_all);
    Y_all = (Y_all - Y_mean) ./ Y_std;

    validIdx = ~any(isnan(X_all), 2) & ~isnan(Y_all);
    X_all = X_all(validIdx, :);
    Y_all = Y_all(validIdx);

    save('MLP_NormalizationParameters.mat', 'X_mean', 'X_std', 'Y_mean', 'Y_std');

    trainRatio = 0.8;
    numTrain = floor(trainRatio * size(X_all, 1));

    X_train = X_all(1:numTrain, :);
    Y_train = Y_all(1:numTrain, :);
    X_test = X_all(numTrain+1:end, :);
    Y_test = Y_all(numTrain+1:end, :);

    numFeatures = numDays * 2 + 1;
    numResponses = 1;

    layers = [
        featureInputLayer(numFeatures)
        fullyConnectedLayer(100)
        reluLayer
        dropoutLayer(0.0)
        fullyConnectedLayer(50)
        reluLayer
        fullyConnectedLayer(numResponses)
        regressionLayer
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 1500, ...
        'MiniBatchSize', min(66, numTrain), ...
        'InitialLearnRate', 0.005, ...
        'GradientThreshold', 1, ...
        'Plots', 'training-progress', ...
        'Verbose', 0);

    net = trainNetwork(X_train, Y_train, layers, options);
    save(networkFileName, 'net');
    disp('网络训练完成，已保存到文件 MLP_trained_model.mat');

    YPred = predict(net, X_test);
    YPred = YPred * Y_std + Y_mean;
    Y_test = Y_test * Y_std + Y_mean;

    mseValue = mean((YPred - Y_test).^2);
    disp(['测试集均方误差（MSE）：', num2str(mseValue)]);

    figure;
    plot(Y_test, 'b-', 'LineWidth', 1.5); hold on;
    plot(YPred, 'r--', 'LineWidth', 1.5);
    legend('真实值', '预测值');
    title('真实值与预测值对比');
    xlabel('样本');
    ylabel('水位值');
    grid on;
end
