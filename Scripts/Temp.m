clc
clearvars
warning off

%% Data analysis

load ..\Data\'Processed data'\Iris_DS.mat

Y = table2array(iris_DS(:, 1:3));
X = table2array(iris_DS(:, 4:end));

B2 = pls(X, Y, false);
Y_hat = normalize(X)*B2;
scatter3(Y_hat(:,2), Y_hat(:,3), Y_hat(:,1))
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

function B = pls(X, Y, mod2, stand, maxIter, tol)
    
    function B1 = pls1(X, Y, maxIter, tol)
        nY = size(Y, 1);
        pY = size(Y, 2);
        mX = size(X, 2);
        maxRank = min(mX, nY);
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
   
    function B2 = pls2(X, Y, maxIter, tol)
        nY = size(Y, 1);
        pY = size(Y, 2);
        mX = size(X, 2);
        maxRank = min(mX, nY);
        E = X; % residual matrix for X
        F = Y; % residual matrix for Y
        for j = 1 : maxRank
            u = F(:, randsample(pY, 1));
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
            mod2 = true;
            stand = true;
            maxIter = 10000;
            tol = 1e-0;
        case 3
            stand = true;
            maxIter = 10000;
            tol = 1e-9;
        case 4
            maxIter = 10000;
            tol = 1e-9;
        case 5
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
        B = pls2(X, Y, maxIter, tol);
    else
        B = pls1(X, Y, maxIter, tol);
    end
end