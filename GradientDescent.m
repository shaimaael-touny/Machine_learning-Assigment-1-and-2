function [theta, Jhistory]= GradientDescent(X, y, theta,alpha, iterations)

    m= length(y);
   
    Jhistory= zeros(iterations,1);

    for i=1:iterations,
        h= signoid( X * theta); %hypothesis vector
        theta= theta-  (alpha * (1/m))* transp(X)* ( (h-y) );
        
  
       Jhistory(i)= calculateCost(X,y,theta); %thetas are updated
    end
       
end

    