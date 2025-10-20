%% ARIMA(p,d,q) 建模与预测（无工具箱，含自动选型/常数项/正确方差）— Final
clear; clc; close all;

%% 1) 生成或导入原始序列（你也可以把这里替换成自己的 y）
rng(42);
n = 500;
AR_coef = 0.7; MA_coef = 0.5;
e = randn(n,1);
y = zeros(n,1); y(1)=e(1);
for t=2:n, y(t)=AR_coef*y(t-1)+e(t)+MA_coef*e(t-1); end

fprintf('========== ARIMA 自动建模 ==========\n\n');
fprintf('样本长度: %d, 均值: %.4f, 标准差: %.4f\n\n', length(y), mean(y), std(y));

figure('Position',[80,80,1200,780]);
subplot(3,2,1); plot(y,'b','LineWidth',1.2); grid on;
title('原始时间序列','FontSize',12); xlabel('时间'); ylabel('数值');

%% 2) 自动定阶：在 d∈{0,1,2}、p,q∈[0,5] 上最小化 AIC
max_p=5; max_q=5; d_candidates=0:2;
best = struct('aic',inf,'p',0,'q',0,'d',0,'ar',[],'ma',[],'c',0,'sigma2',NaN,'logL',-inf);

for d = d_candidates
    yd = difference_series(y,d);       % 差分 d 次
    for p = 0:max_p
        for q = 0:max_q
            try
                % d=0 时估计均值；d=1 时估计漂移，可显著改善长期预测
                include_const = (d<=1);
                [phi, theta, c, sigma2, logL] = fit_arma(yd, p, q, include_const);
                k = p + q + (include_const~=0);
                aic = -2*logL + 2*k;
                if aic < best.aic && isfinite(aic)
                    best = struct('aic',aic,'p',p,'q',q,'d',d, ...
                        'ar',phi,'ma',theta,'c',c,'sigma2',sigma2,'logL',logL);
                end
            catch
                % 忽略失败组合
            end
        end
    end
end

p=best.p; q=best.q; d=best.d;
ar_coef=best.ar; ma_coef=best.ma; c=best.c; sigma2=best.sigma2;

fprintf('最优模型：ARIMA(%d,%d,%d)\n', p,d,q);
fprintf('AIC=%.3f, logL=%.3f, sigma^2=%.4f, 常数项 c=%.4f\n\n', best.aic, best.logL, sigma2, c);

%% 3) ACF / PACF（在最优差分尺度上）
yd = difference_series(y,d);
max_lag=20;
acf_vals = compute_acf(yd,max_lag);
pacf_vals = compute_pacf(yd,max_lag);
subplot(3,2,3);
stem(0:max_lag,acf_vals,'filled'); grid on; title('ACF (差分尺度)','FontSize',12);
yline([1 -1]*1.96/sqrt(length(yd)),'r--'); xlabel('滞后'); ylabel('ACF');
subplot(3,2,4);
stem(1:max_lag,pacf_vals(2:end),'filled'); grid on; title('PACF (差分尺度)','FontSize',12);
yline([1 -1]*1.96/sqrt(length(yd)),'r--'); xlabel('滞后'); ylabel('PACF');

%% 4) 残差与白噪声检验（差分尺度）
residuals = compute_residuals_const(yd, ar_coef, ma_coef, c);
subplot(3,2,2);
plot(yd,'r','LineWidth',1.1); grid on;
title(sprintf('平稳化数据 d=%d',d),'FontSize',12); xlabel('时间'); ylabel('数值');

subplot(3,2,5); plot(residuals,'k','LineWidth',1); grid on;
title('残差序列','FontSize',12); xlabel('时间'); ylabel('残差');

subplot(3,2,6);
histogram(residuals,20,'Normalization','pdf'); hold on;
xx=linspace(mean(residuals)-3*std(residuals),mean(residuals)+3*std(residuals),200);
plot(xx,normpdf(xx,mean(residuals),std(residuals)),'r','LineWidth',1.5);
grid on; title('残差分布','FontSize',12);

res_acf = compute_acf(residuals, 15);
lb_stat = length(residuals)*sum(res_acf(2:end).^2);
try pval = 1-chi2cdf(lb_stat,15); catch, pval = gammainc(lb_stat/2,15/2,'upper'); end
fprintf('Ljung-Box: 统计量=%.3f, p=%.4f  (p>0.05 ⇒ 残差近似白噪声)\n\n', lb_stat, pval);

%% 5) 预测（差分尺度 -> 原尺度），步长=20
forecast_steps = 20;
[fc_diff, ci_diff] = predict_arma_const(yd, ar_coef, ma_coef, c, forecast_steps, sigma2); % 差分尺度
[fc_level, lo_level, hi_level] = undifference_forecast(y, fc_diff, ci_diff, d);          % 还原到原尺度

% 展示预测数值
fprintf('未来 %d 步（原尺度）预测：\n', forecast_steps);
fprintf('步\t点预测\t\t95%%下界\t\t95%%上界\n');
for h=1:forecast_steps
    fprintf('%2d\t% .4f\t% .4f\t% .4f\n', h, fc_level(h), lo_level(h), hi_level(h));
end

%% 6) 作图（原尺度）
figure('Position',[90,90,1000,600]);
plot(1:length(y),y,'b-','LineWidth',1.8,'DisplayName','原始数据'); hold on;
tf = length(y)+(1:forecast_steps);
plot(tf,fc_level,'ro-','LineWidth',1.8,'MarkerSize',6,'DisplayName','预测值');
fill([tf, fliplr(tf)], [lo_level', fliplr(hi_level')], ...
    'r','FaceAlpha',0.20,'EdgeColor','none','DisplayName','95%置信区间');
legend('Location','best'); grid on;
xlabel('时间'); ylabel('数值');
title(sprintf('ARIMA(%d,%d,%d) 模型预测结果（原尺度）',p,d,q),'FontSize',14);
xlim([length(y)-min(200,length(y)-1), length(y)+forecast_steps]);

fprintf('========== 结束 ==========\n');

%% ================= 辅助函数 =================

function yd = difference_series(y,d)
    y = y(:);
    if d==0, yd=y; return; end
    yd = y;
    for k=1:d, yd = diff(yd); end
end

function [fc_level, lo_level, hi_level] = undifference_forecast(y, fc_diff, ci_diff, d)
    % 把差分尺度预测还原到原始尺度；支持 d=0,1,2
    y = y(:);
    switch d
        case 0
            fc_level = fc_diff;
            lo_level = ci_diff(:,1); hi_level = ci_diff(:,2);
        case 1
            base = y(end);
            fc_level = base + cumsum(fc_diff);
            lo_level = base + cumsum(ci_diff(:,1));
            hi_level = base + cumsum(ci_diff(:,2));
        case 2
            base1 = y(end);     % y_T
            base2 = y(end) - y(end-1); % Δy_T
            % 第一步开始：y_{T+h} = y_T + h*Δy_T + cumsum( cumsum(Δ^2 预测) )
            fc_level = zeros(length(fc_diff),1);
            lo_level = fc_level; hi_level = fc_level;
            s_fc = cumsum(fc_diff);     % 累Δ^2 → Δ
            s_lo = cumsum(ci_diff(:,1));
            s_hi = cumsum(ci_diff(:,2));
            fc_level = base1 + (1:length(fc_diff))'*base2 + cumsum(s_fc);
            lo_level = base1 + (1:length(fc_diff))'*base2 + cumsum(s_lo);
            hi_level = base1 + (1:length(fc_diff))'*base2 + cumsum(s_hi);
        otherwise
            error('只实现了 d=0/1/2 的还原');
    end
end

% ========== ACF / PACF ==========
function acf_vals = compute_acf(x, max_lag)
    x = x(:) - mean(x);
    c0 = (x'*x) / length(x);
    acf_vals = zeros(max_lag+1,1); acf_vals(1)=1;
    for k=1:max_lag
        ck = (x(1:end-k)'*x(1+k:end))/length(x);
        acf_vals(k+1) = ck / c0;
    end
end

function pacf_vals = compute_pacf(x, max_lag)
    acf_vals = compute_acf(x, max_lag);
    pacf_vals = zeros(max_lag+1,1); pacf_vals(1)=1; pacf_vals(2)=acf_vals(2);
    for k=2:max_lag
        num = acf_vals(k+1);
        for j=1:k-1, num = num - pacf_vals(j+1)*acf_vals(k-j+1); end
        den = 1;
        for j=1:k-1, den = den - pacf_vals(j+1)*acf_vals(j+1); end
        pacf_vals(k+1) = num / max(den,1e-8);
    end
end

% ========== 拟合：带常数项 c、tanh 约束 ==========
function [phi, theta, c, sigma2, logL] = fit_arma(y, p, q, include_const)
    y = y(:); n = length(y);
    if p==0 && q==0
        c = include_const * mean(y);
        res = y - c; sigma2 = max(var(res),1e-8);
        logL = -n/2*log(2*pi*sigma2) - sum(res.^2)/(2*sigma2);
        phi=[]; theta=[]; return;
    end
    k = p+q + (include_const~=0);
    if include_const
        base_c = mean(y);
    else
        base_c = 0;
    end
    % -------- 多起点优化：显著降低陷入局部最优的概率 --------
    num_starts = max(5, 2*(p+q));
    options = optimset('MaxIter',4000,'Display','off','TolFun',1e-7,'TolX',1e-7);
    best = struct('nll',inf,'params',[],'phi',[],'theta',[],'c',0,'sigma2',NaN,'logL',-inf);

    for s = 1:num_starts
        init = zeros(k,1);
        if p+q>0
            init(1:p+q) = 0.6*randn(p+q,1);
        end
        if include_const
            init(end) = base_c * (1 + 0.2*randn);
        end
        if s==1
            % 第一次使用更保守的初始值（更接近 0）
            if p+q>0, init(1:p+q) = 0.1*randn(p+q,1); end
            if include_const, init(end) = base_c; end
        end

        params = fminsearch(@(w) nll_wrap(y,w,p,q,include_const), init, options);
        nll = nll_wrap(y, params, p, q, include_const);
        if ~isfinite(nll), continue; end
        [phi_tmp,theta_tmp,c_tmp,~,sigma2_tmp,logL_tmp] = unpack_and_eval(y,params,p,q,include_const);
        if ~all(isfinite([phi_tmp(:); theta_tmp(:); c_tmp; sigma2_tmp; logL_tmp]))
            continue;
        end
        if nll < best.nll
            best = struct('nll',nll,'params',params,'phi',phi_tmp,'theta',theta_tmp, ...
                'c',c_tmp,'sigma2',sigma2_tmp,'logL',logL_tmp);
        end
    end

    if ~isfinite(best.nll)
        error('ARMA 优化未收敛，尝试减少 p/q 或检查数据');
    end

    phi = best.phi; theta = best.theta; c = best.c;
    sigma2 = best.sigma2; logL = best.logL;
end

function nll = nll_wrap(y, w, p, q, include_const)
    try
        [~,~,~,res,sigma2,~] = unpack_and_eval(y,w,p,q,include_const);
        n = length(y);
        nll = n/2*log(2*pi*sigma2) + sum(res.^2)/(2*sigma2);
        if ~isfinite(nll), nll=1e12; end
    catch
        nll = 1e12;
    end
end

function [phi,theta,c,residuals,sigma2,logL] = unpack_and_eval(y,w,p,q,include_const)
    % tanh 变换把 AR/MA 系数限制在 (-0.99,0.99)，提高稳定性
    ptr = 0;
    raw_phi = w(1:p);                    ptr=ptr+p;
    raw_theta = w(ptr+1:ptr+q);          ptr=ptr+q;
    phi   = 0.99*tanh(raw_phi(:)).';
    theta = 0.99*tanh(raw_theta(:)).';
    if include_const, c = w(end); else, c = 0; end
    residuals = compute_residuals_const(y, phi, theta, c);
    sigma2 = max(sum(residuals.^2)/length(y), 1e-8);
    logL = -length(y)/2*log(2*pi*sigma2) - sum(residuals.^2)/(2*sigma2);
end

% ========== 残差（带常数项） ==========
function res = compute_residuals_const(y, phi, theta, c)
    y=y(:); n=length(y); p=length(phi); q=length(theta);
    res=zeros(n,1); e=zeros(n,1);
    for t=1:n
        ar=0; for i=1:p, if t-i>0, ar=ar+phi(i)*y(t-i); end, end
        ma=0; for j=1:q, if t-j>0, ma=ma+theta(j)*e(t-j); end, end
        e(t) = y(t) - (c + ar + ma);
        res(t)=e(t);
    end
end

% ========== 预测（带常数 + 历史 MA 残差 + ψ-权方差） ==========
function [forecast, forecast_ci] = predict_arma_const(y, phi, theta, c, steps, sigma2)
    y=y(:); p=length(phi); q=length(theta);
    e_hist = compute_residuals_const(y, phi, theta, c);
    T = length(y);
    forecast = zeros(steps,1);
    y_ext = [y; zeros(steps,1)];
    e_ext = [e_hist; zeros(steps,1)];    % 未来创新期望为 0

    % 点预测：1..q 步带入"可见历史残差"
    for h=1:steps
        t = T+h;
        ar=0; for i=1:p, ar=ar+phi(i)*y_ext(t-i); end
        ma=0; for j=1:q
            idx=t-j;
            if idx>=1 && idx<=T, ma=ma+theta(j)*e_ext(idx); end
        end
        y_ext(t) = c + ar + ma;
        forecast(h) = y_ext(t);
    end

    % 方差：ψ-权累积
    psi = compute_psi_weights(phi, theta, steps-1);
    fvar = zeros(steps,1);
    for h=1:steps, fvar(h)=sigma2*sum(psi(1:h).^2); end
    z=1.96; forecast_ci=[forecast - z*sqrt(fvar), forecast + z*sqrt(fvar)];
end

function psi = compute_psi_weights(phi, theta, K)
    p=length(phi); q=length(theta);
    psi=zeros(K+1,1); psi(1)=1;
    for k=1:K
        val = 0; if k<=q, val = theta(k); end
        for i=1:p, if k-i>=0, val = val + phi(i)*psi(k-i+1); end, end
        psi(k+1)=val;
    end
end
