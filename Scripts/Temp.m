clc
clearvars
warning off

%% Data analysis

data = readtable("..\Data\Sources\stell-faults.csv");
data = removevars(data);

Y = table2array(data(1:739, 28:30));
X = table2array(data(1:739, 1:27));

B2 = pls(X, Y, 26, false);
Y_hat = normalize(X)*B2;
scatter3(Y_hat(1:158,1), Y_hat(1:158,2), Y_hat(1:158,3), 'magenta')
hold on
scatter3(Y_hat(159:348,1), Y_hat(159:348,2), Y_hat(159:348,3), 'yellow')
hold on
scatter3(Y_hat(349:end,1), Y_hat(349:end,2), Y_hat(349:end,3), 'green')
grid on

for i = 1:size(Y_hat, 1)
    [~, j] = max(Y_hat(i, :));
    for k = 1:3
        if k == j
            Y_hat(i, k) = 1;
        else
            Y_hat(i, k) = 0;
        end
    end
end

cont = 0;
for i = 1: size(Y, 1)
    [~, j] = max(Y(i, :));
    [~, k] = max(Y_hat(i, :));
    if j ~= k
        cont = cont + 1;
    end
end

acc = 1 - cont/size(Y, 1);

%% Function

function B = pls(X, Y, alphaRed, mod2, stand, maxIter, tol)
    
    function B1 = pls1(X, Y, alphaRed, maxIter, tol)
        nY = size(Y, 1);
        pY = size(Y, 2);
        mX = size(X, 2);
        maxRank = alphaRed;
        for i = 1 : pY
            f = Y(:, 1);
            y = Y(:, i);
            W = zeros(mX, maxRank);
            P = zeros(mX, maxRank);
            T = zeros(nY, 1);
            E = X;
            for j = 1 : maxRank
                tOld = 0;
                for k = 1 : maxIter
                    w = (E'*y)/norm(E'*y);
                    t = E*w;
                    p = (E'*t)/(t'*t);
                    if abs(tOld - t) < tol
                        break;
                    else
                        tOld = t;
                    end
                end
                
                % scaling
                t = t*norm(p);
                w = w*norm(p);
                p = p/norm(p);
                
                b = (y'*t)/(t'*t);
                E = E - t*p';
                f = f - b*t*1;

                W(:, j) = w;
                P(:, j) = p;
                T(:, j) = t;
            end
            B1(:, i) = W*(P'*W)^-1*(T'*T)^-1*T'*y;
        end
    end
   
    function B2 = pls2(X, Y, alphaRed, maxIter, tol)
        nY = size(Y, 1);
        pY = size(Y, 2);
        mX = size(X, 2);
        maxRank = alphaRed;
        E = X; % residual matrix for X
        F = Y; % residual matrix for Y

        [~, idx] = max(sum(Y.*Y));

        for j = 1 : maxRank
            u = F(:, idx);
            tOld = 0;
            for i = 1 : maxIter
                w = (E'*u)/norm(E'*u); % support vector
                t = E*w; % j-th column of the score matrix for X
                q = (F'*t)/norm(F'*t); % j-th column of the loading matrix for Y
                u = F*q; % j-th column of the score matrix for Y
                if abs(tOld - t) < tol
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
            
            b = (u'*t)/(t'*t); % j-th column of the coefficient regression matrix
            E = E - t*p';
            F = F - b*t*q';

            W(:, j) = w;
            P(:, j) = p;
            T(:, j) = t;

            B2 = W*(P'*W)^-1*(T'*T)^-1*T'*Y;
        end
    end

    switch nargin
        case 2
            alphaRed = max(size(Y, 1), size(X, 2));
            mod2 = true;
            stand = true;
            maxIter = 10000;
            tol = 1e-0;
        case 3
            mod2 = true;
            stand = true;
            maxIter = 10000;
            tol = 1e-9;
        case 4
            stand = true;
            maxIter = 10000;
            tol = 1e-9;
        case 5
            maxIter = 10000;
            tol = 1e-9;
        case 6
            tol = 1e-9;
    end
    
    % data standardization
    if stand
        X = normalize(X);
        Y = normalize(Y);
    end
    
    disp("PLS configuration: ")
    disp("- PLS2: " + mod2);
    disp("- Normalization: " + stand);
    disp("- Maximum iterations: " + maxIter);
    disp("- Tolerance: " + tol);

    if mod2
        B = pls2(X, Y, alphaRed, maxIter, tol);
    else
        B = pls1(X, Y, alphaRed, maxIter, tol);
    end
end