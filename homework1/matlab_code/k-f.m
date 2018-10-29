    %input the train set which is a 20x2 matrix
    Data = [5.16 3.1949;0.67 3.6556;7.96 2.8967;8.52 2.8465;
            9.44 2.6716;1.18 3.3328;0.17 3.9871;3.14 2.3263;
            0.75 3.5996;9.47 2.4775; 4.47 2.8639;9.22 2.5658;
            9.76 2.7043; 1.35 3.1015; 1.86 2.9105; 8.32 2.7196;
            0.53 3.9053;7.78 3.0493; 7.1 3.3612; 8.34 2.8574];
    % get the Data matrix size for future depart.
            [m,n]=size(Data);
% % First part, we need to depart the original train set to 10 parts,
% % and set different parts set as validation sets. In loop function,
% % we can traverse all sets.

%depart train set to 10 parts with the function of crossvalind
indices = crossvalind('Kfold', Data(1:m,n), 10);
%Loop 10 times£¬set the i part as validation set,and set the rest set as train sets
%w = zeros(20,10);
for j = 1:10 
    Test_number = (indices == j);
    Train_number = ~Test_number;
    Train_Data = Data(Train_number,:);
    Test_Data = Data(Test_number,:);
    x = Train_Data(:,1);
    y = Train_Data(:,2);
    [q,~] = size(Train_Data);

    %loop 10 times, store 10 potenial w* value
    w = inv(x*x'+j*I)*x.*y;
    y1 =polyval(w,x);
    x1 = 1:q;
    plot(x,y,'ro');
    x1 = 0:10/q:(10-10/q);
    plot(x1,y1,'b-');
    hold on
    grid on
end