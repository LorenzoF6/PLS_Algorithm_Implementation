clc
clearvars
warning off

%% Metal furnace dataset

alloy_DS_2 = readtable("..\Data\Sources\Train.csv")

n = 5; % number of fault classes
temp = array2table(zeros(size(alloy_DS, 1), n));
temp.Properties.VariableNames = {'class0', 'class1', 'class2', 'class3',...
    'class4'};

for i = 1:size(alloy_DS, 1)
    if alloy_DS.grade(i) == 0
        temp.class0(i) = 1;
    end
    if alloy_DS.grade(i) == 1
        temp.class1(i) = 1;
    end
    if alloy_DS.grade(i) == 2
        temp.class2(i) = 1;
    end
    if alloy_DS.grade(i) == 3
        temp.class3(i) = 1;
    end
    if alloy_DS.grade(i) == 4
        temp.class4(i) = 1;
    end
end

alloy_DS = removevars(alloy_DS ,["grade", "f9"]);
alloy_DS = [temp alloy_DS];

save("..\Data\Processed data\Alloy_DS.mat", "alloy_DS")

%% Iris dataset

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
