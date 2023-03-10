
clc
clearvars
warning off

load ..\Data\'Processed data'\Classes_DS.mat

Y = table2array(classes_DS(:, 1:6));
X = table2array(classes_DS(:, 7:end));

B = pls(X, Y);
% B_2 = PLSI(normalize(X), normalize(Y));
Y_hat = X*B;

for i = 1:size(Y_hat, 1)
    [~, j] = max(Y_hat(i, :));
    for k = 1:6
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

cont/size(Y, 1);
%% 

function B = pls(X, Y, mod2, maxIter, tol)
    
    function B = pls1(X, Y, maxIter, tol)

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
                if tOld - t < tol
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
            maxIter = 10000;
            tol = 10e-10;
        case 3
            maxIter = 10000;
            tol = 10e-10;
        case 4
            tol = 10e-10;
    end
    X = normalize(X);
    % Y = normalize(Y);
    if mod2
        B = pls2(X, Y, maxIter, tol);
    else
        B = pls1(X, Y, maxIter, tol);
    end
end