clear all
load C:\Users\Karma\Desktop\test.txt
% % Homework 1 
% %  abstract: this need us to implement the polynimal curve fitting,
% % and we should use k-fold cross validation to choose the best lambda
% % value. 
% % member: @yuxuan liu @diyang gong @xiaotong li
% % @time : 09/30/2017
p=1;
for i = -50:1:0; %0:0.0005:0.02
   % lambda = 0:19;
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
k=10;%set the k-fold number
%depart train set to 10 parts with the function of crossvalind
indices = crossvalind('Kfold', 20, 10);
%Loop 10 times£¬set the i part as validation set,and set the rest set as train sets
%w = zeros(20,10);
lambda = exp(i);
for j = 1:k 
    Test_number = (indices == j);
    Train_number = ~Test_number;
    Train_Data = Data(Train_number, :);
    Test_Data = Data(Test_number, :);
    x = Train_Data(:,1);
    y = Train_Data(:,2);
    x2 = Test_Data(:,1);
    x3 = zeros(18,10);
    y2 = Test_Data(:,2);
    %[q,~] = size(Train_Data);
     for e = 1:18
        x3(e,:) = x(e,1);
     end
     for u = 1:18
         x3(u,1)=ones(1);
     end
     x6 =zeros(18,10);
     x4 = fliplr(x3); 
    for u = 0:9
        x6(:,u+1)=(x3(:,u+1)).^(u);
    end
    
    %loop 10 times, store 10 potenial w* value
     w(:,p) =normalEqu(x6,y,10,lambda);% pinv(x*x'+j*I)*x.*y;
%    w= polyfit(x,y,10);
   
    
    % y1=0.0001710*x1.^9 - 0.0077813* x1.^8 + 0.14722* x1.^7 - 1.4962* x1.^6 + 8.7941* x1.^5
   %- 29.9208* x1.^4 + 55.4869* x1.^3 - 47.6998* x1.^2 + 11.3714* x1 + 3.7276
   
%     legend('train sets','result');
%     xlabel('Data of x');
%     ylabel('Data of y');
%     hold on
%     grid on
    J_CV(j,p) = costFun(x2,y2,w(:,p))+lambda/2*w(:,p)'*w(:,p);
    MSE(p) = (1/k)*sum(J_CV(:,p));
    RMS(p)= sqrt(2*MSE(p)/2);
    x1 = 0:0.1:10;
  %y1 = polyval(flipud(w(:,p)),x1); 
 
  
end
 p=p+1;
%CV_lambda(i) = 
%plot(lambda,CV_lambda,'rx')
 m = min(MSE);
 mm = min(m);
 [row,column]=find(MSE==mm);
 minErr = MSE(row,column);
 %Best_lambda = 0.0005*column;


 
% xlabel('ln(lambda)');
% ylabel('minErr');
% title('the relationship between Ln(lambda) and minErr');
end
% p2=1;
% for i2=-50:1:-10;
%     lambda3=exp(i2);
%     J_CV2(p2) = costFun(Data(:,1),Data(:,2),w(:,column))+lambda3/2*w(:,column)'*w(:,column);
%     p2=p2+1;
% end    


    
y1 = polyval(flipud(w(:,column)),x1);
figure(2);
plot(test(:,1),test(:,2),'go');
 hold on;
 plot(x1,y1,'b');
  hold on;
  plot(Data(:,1),Data(:,2),'rx');
  hold on;
  plot(x,y,'x');
   figure(1);
   i3=-50:1:0;
   plot(i3,MSE,'r');
% i1=-200:1:-20;
% lambda2=exp(i1);
% 
%  plot(i1,MSE,'r');
% x8=0:0.1:10;
%      y8=(9.73335540297469e-06)*x8.^9 - 0.00042747208135267* x8.^8 + 0.00777040119761112* x8.^7 - 0.0755327208568537* x8.^6 + 0.425054593578939* x8.^5- 1.41403677931492* x8.^4 + 2.75995559551829* x8.^3 -2.89720504728437* x8.^2 + 0.638998005945646* x8 +3.96892243242109;
    