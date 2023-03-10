
clc
clearvars
warning off

classes_DS = readtable("Sources\Fault_classes_data.csv");
n = 6; % number of fault classes
temp = array2table(zeros(size(classes_DS, 1), n));
temp.Properties.VariableNames = {'class1', 'class2', 'class3', 'class4',...
    'class5', 'class6'};

for i = 1:size(classes_DS, 1)
    % class 1: fault between phase A and ground
    if classes_DS.G(i) == 1 && classes_DS.C(i) == 0 &&... 
        classes_DS.B(i) == 0 && classes_DS.A(i) == 1
        temp.class1(i) = 1;
    end

    % class 2: fault between phases A, B and ground
    if classes_DS.G(i) == 1 && classes_DS.C(i) == 0 &&... 
        classes_DS.B(i) == 1 && classes_DS.A(i) == 1
        temp.class2(i) = 1;
    end

    % class 3: fault between phases B and C
    if classes_DS.G(i) == 0 && classes_DS.C(i) == 1 &&... 
        classes_DS.B(i) == 1 && classes_DS.A(i) == 0
        temp.class3(i) = 1;
    end

    % class 4: fault between all three phases
    if classes_DS.G(i) == 0 && classes_DS.C(i) == 1 &&... 
        classes_DS.B(i) == 1 && classes_DS.A(i) == 1
        temp.class4(i) = 1;
    end

    % class 5: three phases symmetrical fault 
    if classes_DS.G(i) == 1 && classes_DS.C(i) == 1 &&... 
        classes_DS.B(i) == 1 && classes_DS.A(i) == 1
        temp.class5(i) = 1;
    end

    % class 6: no fault
    if classes_DS.G(i) == 0 && classes_DS.C(i) == 0 &&... 
        classes_DS.B(i) == 0 && classes_DS.A(i) == 0
        temp.class6(i) = 1;
    end
end

classes_DS = removevars(classes_DS ,["A", "B", "C", "G"]);
classes_DS = [temp classes_DS];

% save("Processed data\Classes_DS.mat", "classes_DS")
