clear all
load C:\Users\Karma\Desktop\test.txt
% % Homework 1 
% %  abstract: this need us to implement the polynimal curve fitting,
% % and we should use k-fold cross validation to choose the best lambda
% % value. 
% % member: @yuxuan liu @diyang gong @xiaotong li
% % @time : 09/30/2017
indices = crossvalind('Kfold', 20, 10);
FS=-10;
FF=0;
FE=0.1;
p=1;
x1 = 0:0.1:10;   %input the train set which is a 20x2 matrix
    Data = [3.5911 2.5747;
4.0993 2.7473;
2.096 2.3768;
5.6045 3.8926;
3.093 2.3875;
8.6065 2.4677;
9.1039 2.9036;
7.1187 3.5931;
9.5946 2.8895;
8.0932 3.5008;
0.60837 3.4854;
5.0971 3.1699;
2.5997 2.4663;
6.5918 3.5727;
1.1036 2.8911;
6.1175 3.5885;
7.6013 2.9082;
1.5966 3.0388;
0.093841 4.0116;
4.5961 3.081];
Data=sortrows(Data,1);
for i = FS:FE:FF;   
  
    % get the Data matrix size for future depart.
            [m,n]=size(Data);
% % First part, we need to depart the original train set to 10 parts,
% % and set different parts set as validation sets. In loop function,
% % we can traverse all sets.
%depart train set to 10 parts with the function of crossvalind
%Loop 10 times£¬set the i part as validation set,and set the rest set as train sets
k=10;%set the k-fold number
lambda =exp(i);
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
        for e = 1:18
            x3(e,:) = x(e,1);
        end
        for u = 1:18
            x3(u,1)=ones(1);
        end
        x6 =zeros(18,10);
        for u = 0:9
            x6(:,u+1)=(x3(:,u+1)).^(u);
        end
        w(:,j) =normalEqu(x6,y,10,lambda);
%       y3 = polyval(flipud(w(:,j)),x1);
%       plot(x1,y3,'k');
% hold on;
        J_CV(j) = costFun(x2,y2,w(:,j))+lambda/2*w(:,j)'*w(:,j);
        J_CV2(j) = costFun(x,y,w(:,j))+lambda/2*w(:,j)'*w(:,j);
        MSE = (1/k)*sum(J_CV);
        MSE_T = (1/k)*sum(J_CV2);
        RMS= sqrt(2*MSE/2);
    end
    D{p}=w;
    J1(:,p)=J_CV;
    MSE1(p)=MSE;
    RMS1(p)=RMS;
    MSE_T1(p)=MSE_T;
    p=p+1;
end

m = min(MSE1);
mm = min(m);
[row,column]=find(MSE1==mm);
minErr = MSE1(row,column);
Bestlambda= exp(FS+column.*0.1);
mmm=min(J1(:,column));
[row1,column1]=find(J1==mmm);
A=D{column};
wx=A(:,row1);

Data11=sort(Data(:,1));
%y1 = polyval(flipud(wx),x1);
y3 = polyval(flipud(wx),Data(:,1));
for ii=1:10
    JJJ=polyval(flipud(A(:,ii)),Data(:,1));
    JJ=(JJJ-y3);
EB(:,ii)=JJ;
end

for iii=1:20
MX= min(EB(iii,:));
MXBR(iii)=MX;
end
v=1;
for iiii=FS:FE:FF
MX1=max(J1(:,v));
MX2=min(J1(:,v));
MXX(v)=MX2;
v=v+1;
end

xx=Data(:,1);
xx3 = zeros(20,10);
 for e = 1:20
            xx3(e,:) = xx(e,1);
        end
        for u = 1:20
            xx3(u,1)=ones(1);
        end
        xx6 =zeros(20,10);
        for u = 0:9
            xx6(:,u+1)=(xx3(:,u+1)).^(u);
        end
        
        
        yd=Data(:,2);
        w1 =normalEqu(xx6,Data(:,2),10,Bestlambda);
        y1 = polyval(flipud(w1),x1);

figure(1);
plot(test(:,1),test(:,2),'go');
hold on;
plot(x1,y1,'m');
hold on;
plot(Data(:,1),Data(:,2),'bx');
hold on;
plot(x,y,'x');
figure(2);
i3=FS:FE:FF;


plot(i3,MSE_T1,'r');
hold on;
errorbar(i3,MSE1,MXX);
hold on;
plot(i3,MSE1,'y');
figure (3);
plot(x1,y1,'m');
figure (4);
x4=Data11;
y4=polyval(flipud(w1),Data11);
%MXBR=[0.0255281221153179,0.0853738947043130,0.137690340675564,0.122225189260698,0.131923714267608,0.0907429857325099,0.0803147932513029,0.0806416537476311,0.0814326923997197,0.0908825298598179,0.103589954223305,0.0953303925226492,0.0681814439205062,0.104034674938793,0.167410746844097,0.190046977021664,0.233945594052397,0.327924587425899,0.193014654706015,0.0100000000000000]
errorbar(x4,y4,MXBR);

w00 =normalEqu(xx6,yd,10,-10);
w000 =normalEqu(xx6,yd,10,exp(5));
y8 = polyval(flipud(w00),x1);
y9 = polyval(flipud(w000),x1);

 figure (5); 
  plot(x1,y1,'m'); 
  hold on; 
  plot(x1,y9,'b');
  hold on;
 plot(x1,y8,'r');


    