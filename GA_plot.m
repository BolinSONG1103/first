clc; clear; close all; rng(42);

% ---------------------- 目标函数定义 ----------------------
% 原函数 f(x1,x2)
f_real = @(x) x(1).^2 + x(2).^2 + 10*sin(x(1)) + 5*sin(x(2));
% 遗传算法仍然执行最小化 → 取负号实现最大化
fitness = @(x) -f_real(x);

% 约束条件：x1 + x2 = 1, 0 ≤ xi ≤ 1
nvars = 2;
A = []; b = [];
Aeq = [1 1]; beq = 1;
lb = [0 0]; ub = [1 1];
nonlcon = [];

% ---------------------- 自定义绘图函数 ----------------------
% gaplotbestf 只能显示最优值，我们添加平均值曲线
plotFcn1 = @(options,state,flag) myGaplot(options,state,flag);

opts = optimoptions('ga',...
    'PopulationSize',60,...
    'MaxGenerations',100,...
    'PlotFcn',{plotFcn1},...
    'Display','iter');

% ---------------------- 运行 GA ----------------------
[x_opt, fval_neg, exitflag, output, population, scores] = ...
    ga(fitness, nvars, A,b,Aeq,beq,lb,ub,nonlcon,opts);

fval = -fval_neg;
fprintf('\n最大值点: x = [%.4f, %.4f], 最大值 f(x) = %.4f\n', x_opt(1), x_opt(2), fval);

% ---------------------- 绘制最终空间分布 ----------------------
[X1,X2] = meshgrid(linspace(0,1,200), linspace(0,1,200));
mask = X1 + X2 <= 1; % 可行域
F = nan(size(X1));
for i=1:numel(X1)
    if mask(i)
        F(i) = f_real([X1(i), X2(i)]);
    end
end
figure;
surf(X1,X2,F,'EdgeColor','none'); hold on;
plot3(x_opt(1),x_opt(2),f_real(x_opt),'rp','MarkerSize',12,'LineWidth',2);
title('GA 寻找最大值的结果 (最终最优点)');
xlabel('x_1'); ylabel('x_2'); zlabel('f(x_1,x_2)');
view(45,30); grid on;

% =====================================================
% ============== 自定义绘图函数定义 =====================
% =====================================================
function state = myGaplot(options,state,flag)
    persistent bestHist meanHist genHist
    switch flag
        case 'init'
            bestHist = [];
            meanHist = [];
            genHist  = [];
            figure('Name','GA 进化过程'); hold on; grid on;
            xlabel('Generation'); ylabel('Fitness (负值 = 越小越好)');
            title('每代最优值 & 平均值演化曲线');
        case 'iter'
            % 记录当前代信息
            bestHist(end+1) = min(state.Score);       % 最优适应度（负值）
            meanHist(end+1) = mean(state.Score);      % 平均适应度
            genHist(end+1)  = state.Generation;

            % 动态更新曲线
            cla;
            plot(genHist, -bestHist,'r-','LineWidth',2);  % 取负数 -> 原函数最大值
            plot(genHist, -meanHist,'b--','LineWidth',1.5);
            legend('当前最优值','平均值','Location','best');
            title(sprintf('迭代中...  第 %d 代', state.Generation));
            xlabel('Generation'); ylabel('f(x)');
            drawnow;
        case 'done'
            % 最终固定显示
            plot(genHist, -bestHist,'r-','LineWidth',2);
            plot(genHist, -meanHist,'b--','LineWidth',1.5);
            legend('最优值','平均值','Location','best');
            title('GA 进化完成');
            grid on;
    end
end
