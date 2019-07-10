function [fval] = funcSinkhornSync(currentX, SynthData)

lambda = 300;
I = SynthData.I;
e = length(I);

a = ones(SynthData.numParticles*SynthData.numParticles,1);
b = a;

currentX = [SynthData.Xs{1} currentX];
 
fval = zeros(e,1);
%fval=0;
for k=1:e
    i = I(1, k); j = I(2, k);
    xi = currentX(:,i);
    xj = currentX(:,j);
    r = xi * (1.0./xj');
    x = r(:);
    
    px = SynthData.Ratios{k};
    
    M = pdist2(x(:),px(:),'euclidean');
    %M = M - diag(diag(M));
    
    K = exp(-lambda.*M);
    K(K<1e-100)=1e-100;
    U = K.*M ;
    sinkDist = sinkhornTransport(a,b,K,U,lambda);
    fval(k)=sinkDist;
    %fval=fval+sinkDist;
end
%disp(fval')
end