function  O= testCostfun(x,y,w)
load C:\Users\Karma\Desktop\test.txt;
test_x = test(:,1);
test_y = test(:,2);
m = length(y); % number of training examples
temp1=polyval(flipud(w),x)-y;
temp2=sum(temp1.^2);
O=1/(2*m)*temp2;
end