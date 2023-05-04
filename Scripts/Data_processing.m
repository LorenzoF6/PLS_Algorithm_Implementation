clc
clearvars
warning off

%% Iris dataset processing

iris_DS = readtable("..\Data\Sources\IRIS.csv");
n = 3; % number of fault classes
temp = array2table(zeros(size(iris_DS, 1), n));
temp.Properties.VariableNames = {'class1', 'class2', 'class3'};

for i = 1:size(iris_DS, 1)
    if strcmp(iris_DS.species{i}, 'Iris-setosa')
        temp.class1(i) = 1;
    end
    if strcmp(iris_DS.species{i}, 'Iris-versicolor')
        temp.class2(i) = 1;
    end
    if strcmp(iris_DS.species{i}, 'Iris-virginica')
        temp.class3(i) = 1;
    end
end

iris_DS = removevars(iris_DS ,["species"]);
iris_DS = [temp iris_DS];

save("..\Data\Processed data\Iris_DS.mat", "iris_DS")
