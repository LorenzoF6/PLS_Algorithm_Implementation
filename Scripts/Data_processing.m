clc
clearvars
warning off

%% Fault detection dataset

faults_DS = readtable("..\Data\Sources\Fault_classes_data.csv");
n = 6; % number of fault classes
temp = array2table(zeros(size(faults_DS, 1), n));
temp.Properties.VariableNames = {'class1', 'class2', 'class3', 'class4',...
    'class5', 'class6'};

for i = 1:size(faults_DS, 1)
    % class 1: fault between phase A and ground
    if faults_DS.G(i) == 1 && faults_DS.C(i) == 0 &&... 
        faults_DS.B(i) == 0 && faults_DS.A(i) == 1
        temp.class1(i) = 1;
    end

    % class 2: fault between phases A, B and ground
    if faults_DS.G(i) == 1 && faults_DS.C(i) == 0 &&... 
        faults_DS.B(i) == 1 && faults_DS.A(i) == 1
        temp.class2(i) = 1;
    end

    % class 3: fault between phases B and C
    if faults_DS.G(i) == 0 && faults_DS.C(i) == 1 &&... 
        faults_DS.B(i) == 1 && faults_DS.A(i) == 0
        temp.class3(i) = 1;
    end

    % class 4: fault between all three phases
    if faults_DS.G(i) == 0 && faults_DS.C(i) == 1 &&... 
        faults_DS.B(i) == 1 && faults_DS.A(i) == 1
        temp.class4(i) = 1;
    end

    % class 5: three phases symmetrical fault 
    if faults_DS.G(i) == 1 && faults_DS.C(i) == 1 &&... 
        faults_DS.B(i) == 1 && faults_DS.A(i) == 1
        temp.class5(i) = 1;
    end

    % class 6: no fault
    if faults_DS.G(i) == 0 && faults_DS.C(i) == 0 &&... 
        faults_DS.B(i) == 0 && faults_DS.A(i) == 0
        temp.class6(i) = 1;
    end
end

faults_DS = removevars(faults_DS ,["A", "B", "C", "G"]);
faults_DS = [temp faults_DS];

save("..\Data\Processed data\Faults_DS.mat", "faults_DS")

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