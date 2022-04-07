
%% 1. READ THE DATA
opts = delimitedTextImportOptions("NumVariables", 7);
opts.DataLines = [2, Inf];
opts.Delimiter = ";";
opts.VariableNames = ["Date", "Close", "Open", "Max", "MIn", "Vol", "var"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "categorical", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Vol", "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Date", "InputFormat", "MM/dd/yyyy");
opts = setvaropts(opts, "var", "TrimNonNumeric", true);
opts = setvaropts(opts, "var", "ThousandsSeparator", ",");
% Import the data
SP500Investing = readtable("C:\Users\chris\OneDrive - Universidad Nacional de Colombia\BM UC3m\Year 2\Advanced Financial Statistics\Activities\S&P500-Investing.csv", opts);
clear opts
HistoricalData = SP500Investing;
	
%% 2. Preparing the data
HistoricalData(1,:) = [];
HistoricalData(:,3:7) = [];
log_returns = table((log(HistoricalData.Close(2:end))- log(HistoricalData.Close(1:end-1)))*100);
HistoricalData(length(HistoricalData.Date),:) = [];
Data = [HistoricalData log_returns];
Data.Properties.VariableNames = {'Date' 'Close' 'Ret'};
sq_Ret = table(Data.Ret.^2);
Data = [Data sq_Ret];
Data.Properties.VariableNames = {'Date' 'Close' 'Ret' 'sq_Ret'};

%% 3. Descriptive stats & Autocorrelations
Des_returns =table( [mean(Data.Ret),median(Data.Ret),skewness(Data.Ret),kurtosis(Data.Ret),...
    std(Data.Ret)]);

Des_sq_returns = table([mean(Data.sq_Ret),median(Data.sq_Ret),skewness(Data.sq_Ret),kurtosis(Data.sq_Ret),...
    std(Data.sq_Ret)]);

figure;
subplot(2,1,1)
plot(Data.Ret);
subplot(2,1,2)
plot(Data.sq_Ret);

ACF_ret = autocorr(Data.Ret);
PACF_ret = parcorr(Data.Ret);

ACF_sq_ret = autocorr(Data.sq_Ret);
PACF_sq_ret = parcorr(Data.sq_Ret);

figure;
subplot(2,1,1)
autocorr(Data.Ret)
subplot(2,1,2)
autocorr(Data.sq_Ret)

[cross_corr,pval] = corr(Data.Ret,Data.sq_Ret);

%% 4. Volatility Estimation - Garch(p,q), p=1, q=1

%Set the model
model = garch('GARCHLags',1,'ARCHLags',1);

% De-mean the return series so we make y = e

Data.Ret = Data.Ret - mean(Data.Ret); 
Data.sq_Ret = Data.Ret.^2;

%Estimation of the model
[Est_model,Est_param,logL] = estimate(model,Data.Ret);

%Calculating the Estimated volatility

cond_Vol = sqrt(infer(Est_model,Data.Ret));

figure;
plot(Data.Ret); hold on;
plot(cond_Vol); hold off;

% Calculating the standarized residuals

Data.std_res = Data.Ret./cond_Vol;
Data.std_sq_res = Data.std_res.^2;

plot(Data.std_res);

%Analyzing if there is serial corr in either of the std ret:

figure;
subplot(2,1,1)
autocorr(Data.std_res)
subplot(2,1,2)
autocorr(Data.std_sq_res)

%% 5. Testing using Engle and Ng (1993)

Data.Dmy = Data.Ret;

%Stting the dummies in the DB

for i = 1:length(Data.Dmy)
    if Data.Ret(length(Data.Dmy)+1-i)<0
        Data.Dmy(length(Data.Dmy)-i) = 1;
    else
        Data.Dmy(length(Data.Dmy)-i) = 0;
    end
end

Data.Ret_1Dmy = Data.Ret.* Data.Dmy;
Data.Ret_0Dmy = Data.Ret.*(1-Data.Dmy);

Data(length(Data.Dmy),:) = []; %remove the last obs

% Fit the OLS
model_ols = fitlm([Data.Dmy,Data.Ret_1Dmy,Data.Ret_0Dmy],Data.std_sq_res);

disp(model_ols)
%%

