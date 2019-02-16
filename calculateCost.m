function J= calculateCost(X,y,theta)

m= length(y);

h= signoid(X*theta); %hypothesis function

J= ((-1/m)*(sum(y.*log(h) +((1-y).*log(1-h)))));

end
