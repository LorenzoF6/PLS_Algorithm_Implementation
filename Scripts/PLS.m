classdef PLS
    properties (Access = public)
        % data
        X {mustBeNumeric}
        X_norm {mustBeNumeric}
        Y {mustBeNumeric}
        Y_norm {mustBeNumeric}
        pY {mustBeNumeric}
        nY {mustBeNumeric}
        mX {mustBeNumeric}
        mod2 {mustBeNumericOrLogical}
        normal {mustBeNumericOrLogical}
        maxIter {mustBeInteger, mustBeNonzero, mustBePositive} 
        tol {mustBeNumeric, mustBeNonzero, mustBePositive}
        alpha {mustBeInteger, mustBeNonzero, mustBePositive}
        % results
        B {mustBeNumeric}
        T {mustBeNumeric}
        P {mustBeNumeric}
        X_hat {mustBeNumeric}
        Y_hat {mustBeNumeric}
        Y_hat_bin {mustBeNumeric}
        MCE {mustBeNumeric}
        confMatrix {mustBeNumeric}
        pMCE
        PCA
        TTV
        CV
        orderRed
    end
    
    methods (Access = public)
        % builder
        function obj = PLS(X, Y, mod, normal, maxIter, tol, print)
            % validation of input arguments
            arguments
                X {mustBeNumeric, mustBeNonmissing}
                Y {mustBeNumeric, mustBeNonmissing}
                mod.Algorithm {mustBeMember(mod.Algorithm, ["PLS2", "PLS1"])} = "PLS2"
                normal.Normalize {mustBeMember(normal.Normalize, ["true", "false"])} = "true"
                maxIter.MaxIterations {mustBeInteger, mustBeNonzero, mustBePositive} = 1000
                tol.ExitTolerance {mustBeNumeric, mustBeNonzero, mustBePositive} = 1e-10
                print.Trace {mustBeMember(print.Trace, ["on", "off"])} = "on"
            end
            % setting of object fields
            obj.X = X;
            obj.mX = size(X, 2);
            obj.Y = Y;
            [obj.nY, obj.pY] = size(Y);
            if mod.Algorithm == "PLS2"
                obj.mod2 = true;
            else
                obj.mod2 = false;
            end
            if normal.Normalize == "true"
                obj.normal = true;
                obj.X_norm = normalize(X);
                obj.Y_norm = normalize(Y);
            else
                obj.normal = false;
            end
            obj.maxIter = maxIter.MaxIterations;
            obj.tol = tol.ExitTolerance;
            if print.Trace == "on"
                disp("PLS configuration: ")
                disp("- Algorithm: " + mod.Algorithm);
                disp("- Normalization: " + obj.normal);
                disp("- Maximum iterations: " + obj.maxIter);
                disp("- Exit tolerance: " + obj.tol);
                disp("- Num. of observations: " + obj.nY);
                disp("- Num. of output variables (Y): " + obj.pY);
                disp("- Num. of input variables (X): " + obj.mX);
                disp("- Order reduction: " + obj.alpha);
            end
        end

        function obj = estimate(obj, orderReduction, PCA)
            arguments
                obj {mustBeNonmissing}
                orderReduction {mustBeInteger, mustBeNonzero} = obj.mX;
                PCA.PCA {mustBeMember(PCA.PCA, ["true", "false"])} = "true"
            end
            obj.alpha = orderReduction;
            if obj.mod2
                [obj.B, obj.T, obj.P] = obj.estimatePLS2;
            else
                [obj.B, obj.T, obj.P] = obj.estimatePLS1;
            end
            % estimation of X_hat and Y_hat
            obj.X_hat = obj.T*obj.P';
            if obj.normal
                obj.Y_hat = obj.X_norm*obj.B;
                if PCA.PCA == "true"
                    [obj.PCA.P, obj.PCA.T] = pca(obj.X_norm, "NumComponents", obj.alpha);
                    obj.PCA.X_hat = obj.PCA.T*obj.PCA.P';
                end
            else
                obj.Y_hat = obj.X*obj.B;
                if PCA.PCA == "true"
                    [obj.PCA.P, obj.PCA.T] = pca(obj.X, 'NumComponents', obj.alpha);
                    obj.PCA.X_hat = obj.PCA.T*obj.PCA.P';
                end
            end
        end

        function obj = predict(obj)
            % classify data
            for i = 1:obj.nY
                [~, j] = max(obj.Y_hat(i, :));
                for k = 1:obj.pY
                    if k == j
                        obj.Y_hat_bin(i, k) = 1;
                    else
                        obj.Y_hat_bin(i, k) = 0;
                    end
                end
            end
            % computation of MCE
            cont = 0;
                for i = 1:obj.nY
                    [~, j] = max(obj.Y(i, :));
                    [~, k] = max(obj.Y_hat_bin(i, :));
                    if j ~= k
                        cont = cont + 1;
                    end
                end
            obj.MCE = cont/obj.nY;
            % computation of MCE for each class
            obj.pMCE = array2table(zeros(1, obj.pY));
            obj.pMCE.Properties.VariableNames = repmat("Class", 1, obj.pY) + (1:obj.pY);
            for j = 1:obj.pY
                classCount = 0;
                errorCount = 0;
                for i = 1:obj.nY
                    if obj.Y(i, j) == 1
                        classCount = classCount + 1;
                        if obj.Y(i, j) ~= obj.Y_hat_bin(i, j)
                            errorCount = errorCount + 1;
                        end
                    end
                end
                obj.pMCE(1, j) = {errorCount/classCount};
            end
            % computation of the confusion matrix (entire dataset)
            Y_class = zeros(obj.nY, 1);
            Y_class_hat = zeros(obj.nY, 1);
            for i = 1:obj.nY
                [~, Y_class(i, 1)] = max(obj.Y(i, :));
                [~, Y_class_hat(i, 1)] = max(obj.Y_hat_bin(i, :));
            end
            obj.confMatrix = confusionmat(Y_class, Y_class_hat);
        end

        function obj = validate(obj, testPercent, repeat)
            arguments
                obj {mustBeNonmissing}
                testPercent {mustBeNumeric, mustBeNonzero, mustBePositive, ...
                    mustBeLessThanOrEqual(testPercent, 1)} = .30
                repeat.Repeatable {mustBeMember(repeat.Repeatable, ["true", "false"])} = "false"
            end
            if repeat.Repeatable == "true"
                rng("default");
            end
            obj.TTV.TestPercent = testPercent;
            % definition of data structures
            ValMCE = array2table(zeros(1, 2));
            ValMCE.Properties.VariableNames = ["Train", "Test"];
            ValpMCE = array2table(zeros(2, obj.pY));
            ValpMCE.Properties.VariableNames = repmat("Class", 1, obj.pY) + (1:obj.pY);
            ValpMCE.Properties.RowNames = ["Train", "Test"];
            % computation of train and test data
            testIdx = [];
            for j = 1:obj.pY
                dataClass = find(obj.Y(:, j) == 1);
                testIdx = [testIdx; randsample(dataClass,...
                    round(size(dataClass, 1)*testPercent))];
            end
            trainIdx = setdiff(1:obj.nY, testIdx);
            obj.TTV.X_train = obj.X(trainIdx, :);
            obj.TTV.Y_train = obj.Y(trainIdx, :);
            obj.TTV.X_test = obj.X(testIdx, :);
            obj.TTV.Y_test = obj.Y(testIdx, :);
            % PLS estimation (train)
            if obj.mod2
                obj1 = PLS(obj.TTV.X_train, obj.TTV.Y_train, "Algorithm", "PLS2", "Trace", "off");
            else
                obj1 = PLS(obj.TTV.X_train, obj.TTV.Y_train, "Algorithm", "PLS1", "Trace", "off");
            end
            obj1 = obj1.estimate(obj.alpha);
            obj1 = obj1.predict;
            ValMCE(1, 1) = array2table(obj1.MCE);
            ValpMCE(1, :) = obj1.pMCE;
            % PLS estimation (test)
            if obj.normal
                [ValMCE(1, 2), temp1, temp2] = PLS.predictStatic(...
                    obj.TTV.Y_test, normalize(obj.TTV.X_test)*obj1.B);
            else
                [ValMCE(1, 2), temp1, temp2] = PLS.predictStatic(...
                    obj.TTV.Y_test, obj.TTV.X_test*obj1.B);
            end
            ValpMCE(2, :) = array2table(temp1);
            % results saving
            obj.TTV.MCE = ValMCE;
            obj.TTV.pMCE = ValpMCE;
            obj.TTV.confMatrix = temp2;
        end

        function obj = crossval(obj, kFold, repeat)
            arguments
                obj {mustBeNonmissing}
                kFold {mustBeInteger} = 10;
                repeat.Repeatable {mustBeMember(repeat.Repeatable, ["true", "false"])} = "false"
            end
            if repeat.Repeatable == "true"
                rng("default");
            end
            % setting data structures and parameters
            obj.CV.kFold = kFold;
            idx = randsample(obj.nY, obj.nY, false);
            obj.CV.X_rand = obj.X(idx, :);
            obj.CV.Y_rand = obj.Y(idx,:);
            step = round(obj.nY/kFold);
            start = 1;
            obj.CV.kMCE = array2table(zeros(kFold, 2));
            obj.CV.kMCE.Properties.VariableNames = ["Train", "Test"];
            obj.CV.avg_kMCE = array2table(zeros(1, 2));
            obj.CV.avg_kMCE.Properties.VariableNames = ["Train", "Test"];
            obj.CV.avg_pMCE = array2table(zeros(2, obj.pY));
            obj.CV.avg_pMCE.Properties.VariableNames = repmat("Class", 1, obj.pY) + (1:obj.pY);
            obj.CV.avg_pMCE.Properties.RowNames = ["Train", "Test"];
            temp_train = zeros(1, obj.pY);
            temp_test = zeros(1, obj.pY);
            % execution of cross-validation
            for foldIndex = 1:kFold
                if foldIndex < kFold
                    idx = start:(start + step - 1);
                    start = start + step;
                else
                    idx = start:obj.nY;
                end
                % computation of test data
                X_test = obj.CV.X_rand(idx, :);
                Y_test = obj.CV.Y_rand(idx, :);
                % computation of train data
                resIdx = setdiff(1:obj.nY, idx);
                X_train = obj.CV.X_rand(resIdx, :);
                Y_train = obj.CV.Y_rand(resIdx, :);
                % PLS estimation (train)
                if obj.mod2
                    obj1 = PLS(X_train, Y_train, "Algorithm", "PLS2", "Trace", "off");
                else
                    obj1 = PLS(X_train, Y_train, "Algorithm", "PLS1", "Trace", "off");
                end
                obj1 = obj1.estimate(obj.alpha);
                obj1 = obj1.predict;
                obj.CV.kMCE(foldIndex, 1) = array2table(obj1.MCE);
                temp_train = temp_train + table2array(obj1.pMCE);
                % PLS estimation (test)
                if obj.normal
                    [obj.CV.kMCE(foldIndex, 2), temp] = PLS.predictStatic(...
                        Y_test, normalize(X_test)*obj1.B);
                else
                    [obj.CV.kMCE(foldIndex, 2), temp] = PLS.predictStatic(...
                        Y_test, X_test*obj1.B);
                end
                temp_test = temp_test + temp;
            end
            % results saving
            obj.CV.avg_kMCE(1, 1) = array2table(mean(obj.CV.kMCE{:, 1}));
            obj.CV.avg_kMCE(1, 2) = array2table(mean(obj.CV.kMCE{:, 2}));
            obj.CV.avg_pMCE(1, :) = array2table(temp_train/kFold);
            obj.CV.avg_pMCE(2, :) = array2table(temp_test/kFold);
        end

        function obj = orderAnalysis(obj, nIter, print)
            arguments
                obj {mustBeNonmissing}
                nIter {mustBeInteger, mustBeNonzero, mustBePositive} = 10
                print.Trace {mustBeMember(print.Trace, ["on", "off"])} = "off"
            end
            obj.orderRed.nIter = nIter;
            % calculation of the best alpha for each iteration
            bestAvgMCE = zeros(nIter, 2);
            alphaMCE = zeros(obj.mX, nIter);
            for i = 1:nIter
                alphaAvgMCE_i = computeMCEByOrder(obj, i, nIter, print.Trace);
                [bestMCE_i, bestAlpha_i] = min(alphaAvgMCE_i);
                alphaMCE(:, i) = alphaAvgMCE_i;
                bestAvgMCE(i, :) = [bestMCE_i, bestAlpha_i];
            end
            alphaMCE = array2table(alphaMCE);
            alphaMCE.Properties.VariableNames = repmat("Iter", 1, nIter) + (1:nIter);
            % calculation of some statistics regarding MCE
            MCE_statistics = array2table(zeros(obj.mX, 10));
            MCE_statistics.Properties.VariableNames = ["Min", "Prct25",...
                "Avg", "Median", "Prct75", "Max", "Std", "Skewness",...
                "Kurtosis", "JB"];
            for i = 1:obj.mX
                MCE_statistics(i, 1) = array2table(min(alphaMCE{i, :}));
                MCE_statistics(i, 2) = array2table(prctile(alphaMCE{i, :}, 25));
                MCE_statistics(i, 3) = array2table(mean(alphaMCE{i, :}));
                MCE_statistics(i, 4) = array2table(median(alphaMCE{i, :}));
                MCE_statistics(i, 5) = array2table(prctile(alphaMCE{i, :}, 75));
                MCE_statistics(i, 6) = array2table(max(alphaMCE{i, :}));
                MCE_statistics(i, 7) = array2table(std(alphaMCE{i, :}));
                MCE_statistics(i, 8) = array2table(skewness(alphaMCE{i, :}));
                MCE_statistics(i, 9) = array2table(kurtosis(alphaMCE{i, :}));
                MCE_statistics(i, 9) = array2table(kurtosis(alphaMCE{i, :}));
                MCE_statistics(i, 10) = array2table(jbtest(alphaMCE{i, :}));
            end
            % calculation of the best alpha
            bestAlphaCounters = array2table(zeros(obj.mX, 2));
            bestAlphaCounters.Properties.VariableNames = ["Counter", "avgMCE"];
            for ord = 1:obj.mX
                acc = 0;
                count = 0;
                for i = 1:nIter
                    if bestAvgMCE(i, 2) == ord
                        count = count +1;
                        acc = acc + bestAvgMCE(i, 1);
                    end
                end
                if count ~= 0
                    bestAlphaCounters(ord, :) = {count, acc/count};
                else
                    bestAlphaCounters(ord, :) = {count, 0};
                end
            end
            % results saving
            [~, obj.orderRed.bestAlpha] = max(bestAlphaCounters{:, 1});
            obj.orderRed.bestAlphaCounters = bestAlphaCounters;
            obj.orderRed.alphaMCE = alphaMCE;
            obj.orderRed.MCE_statistics = MCE_statistics;
        end

        function plotP(obj)
            
        end
    end

    methods (Static)
        function [MCE, pMCE, confMatrix] = predictStatic(Y, Y_hat)
            [nY, pY] = size(Y);
            Y_hat_bin = zeros(nY, pY);
            % classify data
            for i = 1:nY
                [~, j] = max(Y_hat(i, :));
                for k = 1:pY
                    if k == j
                        Y_hat_bin(i, k) = 1;
                    else
                        Y_hat_bin(i, k) = 0;
                    end
                end
            end
            % computation of MCE
            cont = 0;
            for i = 1:nY
                [~, j] = max(Y(i, :));
                [~, k] = max(Y_hat_bin(i, :));
                if j ~= k
                    cont = cont + 1;
                end
            end
            MCE = array2table(cont/nY);
            % computation of MCE for each class
            pMCE = zeros(1, pY);
            for j = 1:pY
                classCount = 0;
                errorCount = 0;
                for i = 1:nY
                    if Y(i, j) == 1
                        classCount = classCount + 1;
                        if Y(i, j) ~= Y_hat_bin(i, j)
                            errorCount = errorCount + 1;
                        end
                    end
                end
                pMCE(1, j) = errorCount/classCount;
            end
            % computation of the confusion matrix (entire dataset)
            Y_class = zeros(nY, 1);
            Y_class_hat = zeros(nY, 1);
            for i = 1:nY
                [~, Y_class(i, 1)] = max(Y(i, :));
                [~, Y_class_hat(i, 1)] = max(Y_hat_bin(i, :));
            end
            confMatrix = confusionmat(Y_class, Y_class_hat);
        end
    end

    methods (Access = private)
        function [B2, T, P] = estimatePLS2(obj)
            maxRank = obj.alpha;
            if obj.normal
                X_1 = obj.X_norm;
                Y_1 = obj.Y_norm;
            else
                X_1 = obj.X;
                Y_1 = obj.Y;
            end
            E = X_1; % residual matrix for X
            F = Y_1; % residual matrix for Y
            [~, idx] = max(sum(Y_1.*Y_1));
            % search of the j-th eigenvector
            for j = 1:maxRank
                u = F(:, idx);
                tOld = 0;
                for i = 1:obj.maxIter
                    w = (E'*u)/norm(E'*u); % support vector
                    t = E*w; % j-th column of the score matrix for X
                    q = (F'*t)/norm(F'*t); % j-th column of the loading matrix for Y
                    u = F*q; % j-th column of the score matrix for Y
                    if abs(tOld - t) < obj.tol
                        break;
                    else 
                        tOld = t;
                    end
                end
                p = (E'*t)/(t'*t); % j-th column of the loading matrix of X
                % scaling
                t = t*norm(p);
                w = w*norm(p);
                p = p/norm(p);
                % calculation of the error matrices
                b = (u'*t)/(t'*t); % j-th column of the coefficient regression matrix
                E = E - t*p';
                F = F - b*t*q';
                % calculation of W, P, T and B2
                W(:, j) = w;
                P(:, j) = p;
                T(:, j) = t;
                B2 = W*(P'*W)^-1*(T'*T)^-1*T'*Y_1;
            end
        end

        function [B1, T, P] = estimatePLS1(obj)
            maxRank = obj.alpha;
            if obj.normal
                X_1 = obj.X_norm;
                Y_1 = obj.Y_norm;
            else
                X_1 = obj.X;
                Y_1 = obj.Y;
            end
            % optimizing for each output variable
            for i = 1:obj.pY
                f = Y_1(:, 1);
                y = Y_1(:, i);
                W = zeros(obj.mX, maxRank);
                P = zeros(obj.mX, maxRank);
                T = zeros(obj.nY, 1);
                E = X_1;
                % search of the j-th eigenvector
                for j = 1:maxRank
                    tOld = 0;
                    for k = 1 : obj.maxIter
                        w = (E'*y)/norm(E'*y);
                        t = E*w;
                        p = (E'*t)/(t'*t);
                        if abs(tOld - t) < obj.tol
                            break;
                        else
                            tOld = t;
                        end
                    end
                    % scaling
                    t = t*norm(p);
                    w = w*norm(p);
                    p = p/norm(p);
                    % calculation of the error matrices
                    b = (y'*t)/(t'*t);
                    E = E - t*p';
                    f = f - b*t*1;
                    % calculation of W, P and T
                    W(:, j) = w;
                    P(:, j) = p;
                    T(:, j) = t;
                end
                B1(:, i) = W*(P'*W)^-1*(T'*T)^-1*T'*y;
            end
        end

        function alphaAvgMCE = computeMCEByOrder(obj, iter, nIter, trace)
            alphaAvgMCE = zeros(obj.mX, 1);
            for order = 1:obj.mX
                if trace == "on"
                    percent = round(((order + (iter-1)*obj.mX)/(obj.mX*nIter))*100, 2);
                    disp("Iteration " + iter + " of " + nIter + ", order " +...
                        order + " of " + obj.mX + " (" + percent + "%)");
                end
                if obj.mod2
                    obj1 = PLS(obj.X, obj.Y, "Algorithm", "PLS2", "Trace", "off");
                else
                    obj1 = PLS(obj.X, obj.Y, "Algorithm", "PLS1", "Trace", "off");
                end
                obj1 = obj1.estimate(order);
                obj1 = obj1.predict;
                obj1 = obj1.crossval;
                alphaAvgMCE(order, 1) = table2array(obj1.CV.avg_kMCE(1, 2));
            end
        end
    end
end
