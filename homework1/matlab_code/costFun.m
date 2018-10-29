function J = costFun(x,y,w)
m = length(y); % number of training examples
temp1=polyval(flipud(w),x)-y;
temp2=sum(temp1.^2);
J=(1/(2*m))*temp2;
end