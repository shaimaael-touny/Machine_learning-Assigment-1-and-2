
%hypothesis 1
clear all
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T= read(ds); % table
size(T);
alpha= 0.0001;
n=4;
iterations= 100000;
theta=zeros((n),1);

%training set 60%
U1=T{1:150,1}; % column of age
U2=T{1:150,3}; %column of cp
U3=T{1:150,6}; %column of fbs
y=T{1:150,14}; %column of target

%cv set 20%
U11=T{151:200,1}; % column of age
U22=T{151:200,3}; %column of cp
U33=T{151:200,6}; %column of fbs
y1=T{151:200,14}; %column of target

%test set 20%
U111=T{201:250,1}; % column of age
U222=T{201:250,3}; %column of cp
U333=T{201:250,6}; %column of fbs
y111=T{201:250,14}; %column of target


m=length(y);
mcv= length(y1);
mtest=length(y111);

%cost
X=[ ones(m,1) U1 U2 U3];
XX=[ ones(mcv,1) U11 U22 U33];
XXX=[ ones(mtest,1) U111 U222 U333];

for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));   %scaling of vector x
    end
end




%gradient
[theta, Jhistory] = GradientDescent(X,y,theta,alpha,iterations);

Jcv= calculateCost(XX,y1,theta);
Jtest= calculateCost(XXX,y111,theta);

figure(1)
plot(1:iterations , Jhistory)
xlabel('NumberOfIterations')
ylabel('Error')

