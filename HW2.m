
%% Downloading the data : 03/01/1990 to 25/03/2019
data = getMarketDataViaYahoo('^GSPC', '3-Jan-1990', '25-Mar-2019', '1d');
returns = price2ret(data.AdjClose)*100;

%% EX1: Fit an ARIMA model:

[~, ~, innovations] = armaxfilter(returns,1,1,1);

%% Execute a GRJ-GARCH MODEL(1,1) and a GARCH(1,1) :

[parameters_GGARCH,LL_GGARCH,ht_GGARCH] = tarch(innovations,1,1,1);
[parameters_GARCH,LL_GARCH,ht_GARCH] = tarch(innovations,1,0,1);

subplot(3,1,1), plot(innovations), title('Residuals');
subplot(3,1,2), plot(sqrt(ht_GGARCH)), title('Estimated volatility');
subplot(3,1,3), plot(innovations./sqrt(ht_GGARCH)), title('Standardized residuals');

%% Compare both c.volatilities
figure;
plot(sqrt(ht_GGARCH)); hold on;
plot(sqrt(ht_GARCH)); hold off;
title('Estimated Volatilities between the GARCH and GJR-GARCH')
ylabel('%')
legend('GJR-GARCH','GARCH')

model= gjr(1,1);
[Est_modelGG,Est_paramGG,logLGG] = estimate(model,innovations);

model2= garch(1,1);
[Est_modelG,EstG_paramG,logLG] = estimate(model2,innovations);

%% LL ratio

[LRh,LRp,LRstat,cV] = lratiotest(LL_GGARCH,LL_GARCH,1);

%% AIC & BIC
[aic_GG,bic_GG] = aicbic(logLGG,4,length(innovations));
[aic_G,bic_G] = aicbic(logLG,3,length(innovations));








