%% SSAE (Stacked Sparse Autoencoder)
clear; clc;

% 1) 数据
[XTrain, YTrain] = digitTrain4DArrayData;   % 28x28x1xN
[XTest,  YTest ] = digitTest4DArrayData;

XTrainCol = reshape(XTrain, [], size(XTrain,4));  % 784xN，列为样本
XTestCol  = reshape(XTest,  [], size(XTest,4));

% 2) 第1层稀疏AE：784 -> 256
hiddenSize1 = 256;
autoenc1 = trainAutoencoder( ...
    XTrainCol, hiddenSize1, ...
    'MaxEpochs', 30, ...
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'ScaleData', true);             

% 3) 第2层稀疏AE：256 -> 64
feat1Train = encode(autoenc1, XTrainCol);   % 256xN
hiddenSize2 = 64;
autoenc2 = trainAutoencoder( ...
    feat1Train, hiddenSize2, ...
    'MaxEpochs', 30, ...
    'L2WeightRegularization', 0.001, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', 0.05, ...
    'ScaleData', false);           

% === 4) Softmax 分类头（把标签转为 one-hot） ===
feat2Train = encode(autoenc2, feat1Train);   % 64×N
classes = categories(YTrain);
numClasses = numel(classes);

% N×K -> 转置成 K×N，作为目标 TTrain
TTrain = dummyvar(categorical(YTrain, classes))';   % (numClasses × N)

softnet = trainSoftmaxLayer(feat2Train, TTrain, 'MaxEpochs', 50);

% === 5) 堆叠 + 端到端微调（同样用 one-hot 目标） ===
deepnet = stack(autoenc1, autoenc2, softnet);
deepnet = train(deepnet, XTrainCol, TTrain);  % 注意这里也要用 TTrain

% === 6) 测试 ===
YPred = deepnet(XTestCol);              % 10×N（每列是各类概率）
[~, idx] = max(YPred, [], 1);
YPredLabel = categorical(classes(idx));

acc = mean(YPredLabel == YTest);
fprintf('Test Accuracy = %.2f%%\n', acc*100);


% 7) 可视化第一层特征
figure; plotWeights(autoenc1); title('L1 Filters (W1)');

figure;
W = autoenc1.EncoderWeights;     % 获取权重矩阵 W1
numNeurons = size(W,2);          % 隐藏层神经元数量
numDisplay = min(100,numNeurons); % 最多显示100个

for i = 1:numDisplay
    subplot(10,10,i);
    % 自动推算图像尺寸
    nPixels = size(W,1);
    side = sqrt(nPixels);        % 假设输入为平方图像
    imagesc(reshape(W(:,i), side, side));
    colormap gray;
    axis off;
end
sgtitle('First-layer learned features (auto)');
