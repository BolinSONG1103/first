clc; clear;

f_real = @(x) x(1).^2 + x(2).^2 + 10*sin(x(1)) + 5*sin(x(2));
fitness = @(x) f_real(x);
lb = [-10 -10]; ub = [10 10];
A = []; b = [];
Aeq = [1 1]; beq = 1;
opts = optimoptions('ga',...
    'PopulationSize',50, ...
    'MaxGenerations',100,...
    'EliteCount', 4, ...
    'PlotFcn',{'gaplotbestf'});  % 画出每代最优值

[x_opt, fval] = ga(fitness, 2, A,b,Aeq,beq, lb, ub);
fprintf('最优解 x = [%.4f, %.4f], f(x) = %.4f\n', x_opt(1), x_opt(2), fval);
% 可视化
t = linspace(0,1,201);
X1 = t; X2 = 1-t;
F = f_real([X1; X2]);
figure; plot(X1, F); hold on;
plot(x_opt(1), f_real(x_opt), 'rp', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('x_1'); ylabel('f(x_1, 1-x_1)');
title('GA 寻找最小值结果');
grid on;
