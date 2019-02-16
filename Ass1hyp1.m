
%hypothesis 1 
clear all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=0.001;






%training set 60%
m=length(T{1:10800,1}); %numberofhouses 60% of rows
U1=T{1:10800,4}; % column of bedrooms
U2=T{1:10800,5}; %column of bathrooms
U3=T{1:10800,8}; %column of floors
U4=T{1:10800,12};%coulmn of grade
X=[ones(m,1) U1 U2 U3 U4 ];
n=length(X(1,:)); %number of features
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));   %scaling of vector x
    end
end

%cv set (20%)
mcv= length(T{10801:14401,1});
U11=T{10801:14401,4}; % column of bedrooms
U22=T{10801:14401,5}; %column of bathrooms
U33=T{10801:14401,8}; %column of floors
U44=T{10801:14401,12};%coulmn of grade
XX=[ones(mcv,1) (U11).^3 (U22).^3 (U33).^3 (U44).^3 ];
n2=length(XX(1,:)); %number of features
for w=2:n
    if max(abs(XX(:,w)))~=0
    XX(:,w)=(XX(:,w)-mean((XX(:,w))))./std(XX(:,w));   %scaling of vector x
    end
end

%test set (20%)
mtest= length(T{14402:17999,1});
U111=T{14402:17999,4}; % column of bedrooms
U222=T{14402:17999,5}; %column of bathrooms
U333=T{14402:17999,8}; %column of floors
U444=T{14402:17999,12};%coulmn of grade
XXX=[ones(mtest,1) (U111).^3 (U222).^3 (U333).^3 (U444).^3 ];
n3=length(XXX(1,:)); %number of features
for w=2:n
    if max(abs(XXX(:,w)))~=0
    XXX(:,w)=(XXX(:,w)-mean((XXX(:,w))))./std(XXX(:,w));   %scaling of vector x
    end
end



Y=T{1:10800,3}/mean(T{:,3}); %price after scaling
Y1=T{10801:14401,3}/mean(T{:,3}); %price after scaling of cv
Y11=T{14402:17999,3}/mean(T{:,3}); %price after scaling of test
Theta=zeros(n,1);
k=1;

E(k)=(1/(2*m))*sum((X*Theta-Y).^2);


R=1;
while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.0001;
    R=0;
end
end

E1=(1/(2*mcv))*sum((XX*Theta-Y1).^2); %J of cv
E11=(1/(2*mtest))*sum((XXX*Theta-Y11).^2); %J of test

figure(1)
plot(E)
xlabel('number of iterations');
ylabel('Error');
