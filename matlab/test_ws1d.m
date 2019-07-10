
N = 10;
mu = [1 -1]; 
sigma = [.9 .4; .4 .3]; 
X = mvnrnd(mu,sigma,N); 
p = mvnpdf(X,mu,sigma);

mu2 = [4 -4]; 
sigma2 = [.65 .5; .5 .5]; 
X2 = mvnrnd(mu2,sigma2,N); 
p2 = mvnpdf(X2,mu2,sigma2);

figure, plot(X(:,1),X(:,2),'r+');
hold on, plot(X2(:,1),X2(:,2),'b+');

%R = RADON(I,THETA,N);


