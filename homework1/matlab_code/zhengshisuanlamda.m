for lam1=1:1000
%         lam=exp(0.001*lam1);
    lam=lam1*0.0001;
    n=10;
    E=zeros(1,20);
    EE=1:10;
    X_new=1:18;
    Y_new=1:18;
    
    for i=1:10
        
        for j=1:(2*i-2)
            X_new(j)=X(j);
        end
        for j=(2*i+1):20
            X_new(j-2)=X(j);
        end
        
        for j=1:(2*i-2)
            Y_new(j)=Y(j);
        end
        for j=(2*i+1):20
            Y_new(j-2)=Y(j);
        end
        
        XX=eye(n,18);
        for j=1:n
            XX(j,:)=X_new.^(j-1);
        end
        w=(XX*(XX')+lam*eye(n))\XX*(Y_new');
        
        
        
        for j=(2*i-1):(2*i)
            for p=1:n
                E(j)=E(j)+w(p)*X(j)^(p-1);
            end
            E(j)=(E(j)-Y(j))^2;
        end
        EE(i)=E(2*i-1)+E(2*i)+lam*(w'*w);
    end
    
    EEm(lam1)=mean(EE);
    %     D(lam1)=find(EEm==min(EEm));
end
lam1=1:1000;
D=find(EEm==min(EEm));
plot(lam1,EEm);