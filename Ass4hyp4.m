%hypothesis 4
clear all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
size(T);
Alpha=0.005;

%training set 60% 
m=length(T{1:12964,1}); %numberofhouses(60% of rows)
U1=T{1:12964,4}; % column of bedrooms
U2=T{1:12964,5}; %column of bathrooms
U3=T{1:12964,8}; %column of floors
U4=T{1:12964,12};%coulmn of grade
X=[ones(m,1) U1.^3 U2.^3 U3.^3 U4.^3 ]; %60 %of training set
n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));   %scaling of vector x
    end
end


%cv set (20%)
mcv= length(T{12965:17285,1});
U11=T{12965:17285,4}; % column of bedrooms
U22=T{12965:17285,5}; %column of bathrooms
U33=T{12965:17285,8}; %column of floors
U44=T{12965:17285,12};%coulmn of grade
XX=[ones(mcv,1) (U11).^3 (U22).^3 (U33).^3 (U44).^3 ];
n2=length(XX(1,:)); %number of features
for w=2:n
    if max(abs(XX(:,w)))~=0
    XX(:,w)=(XX(:,w)-mean((XX(:,w))))./std(XX(:,w));   %scaling of vector x
    end
end


Y=T{1:12964,3}/mean(T{:,3}); %price after scaling of 60%
Y1=T{12965:17285,3}/mean(T{:,3}); %price after scaling of cv
Y11=T{17286:21607,3}/mean(T{:,3}); %price after scaling of test
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
if q <.00001;
    R=0;
end
end

E1=(1/(2*mcv))*sum((XX*Theta-Y1).^2); %J of cv



figure(4)
plot(E)
xlabel('number of iterations');
ylabel('ErrorOfTrainingSet');

