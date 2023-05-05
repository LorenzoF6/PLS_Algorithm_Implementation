clc
clearvars
warning off

%% Data import

data = readtable("..\Data\Sources\stell-faults.csv");
faultClasses = data.Properties.VariableNames;
faultClasses = faultClasses(28:end);
rowLimits = [158 348 739 811 866 1268 size(data, 1)];
colLimits = 28:size(data, 2);

%% Models estimation and validation

numIter = 50;
testPercent = .30;

% PLS2
for nVar = 2:length(faultClasses)
    subset = data(1:rowLimits(nVar), 1:colLimits(nVar));
    X = subset{1:rowLimits(nVar), 1:28};
    Y = subset{1:rowLimits(nVar), faultClasses(1:nVar)};
    model = PLS(X, Y, "Algorithm", "PLS2", "Trace", "on");
    model = model.orderAnalysis(numIter, "Trace", "on");
    model = model.estimate(model.orderRed.bestAlpha);
    model = model.predict;
    model = model.validate(testPercent, "Repeatable", "true");
    model = model.crossval("Repeatable", "true");
    models_PLS2.("p" + nVar) = model;
end
save("..\Data\Models\PLS2_hat", "models_PLS2");

% PLS1
for nVar = 2:length(faultClasses)
    subset = data(1:rowLimits(nVar), 1:colLimits(nVar));
    X = subset{1:rowLimits(nVar), 1:28};
    Y = subset{1:rowLimits(nVar), faultClasses(1:nVar)};
    model = PLS(X, Y, "Algorithm", "PLS1", "Trace", "on");
    model = model.orderAnalysis(numIter, "Trace", "on");
    model = model.estimate(model.orderRed.bestAlpha);
    model = model.predict;
    model = model.validate(testPercent, "Repeatable", "true");
    model = model.crossval("Repeatable", "true");
    models_PLS1.("p" + nVar) = model;
end
save("..\Data\Models\PLS1_hat", "models_PLS1");