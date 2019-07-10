
close all;

N = 8;
completeness = 1;
maxNumModes = 4;
numParticles = 16;

disp (['Generating synthetic data. N=' num2str(N), ', #modes=', num2str(maxNumModes), ', numParticles=', numParticles]);
[SynthData] = gen_synth_data_1d(N, completeness, maxNumModes, numParticles);

disp ('Initializing solution');
% initialize solution randomly
N = SynthData.N;
numParticles = SynthData.numParticles;
sol = cell(N, 1);
solY = cell(N, 1);
for i=1:N
    %sol{i} = rand(numParticles, 1);
    sol{i} = SynthData.Xs{i}+0.075*randn(numParticles,1);
    solY{i} = ones(numParticles, 1)./numParticles; % initialize uniform
end

sol{1} = SynthData.Xs{1};
%sol{end} = SynthData.Xs{end};

disp ('Optimize...');
X0 = cell2mat(sol');    
options = optimoptions('lsqnonlin','display','iter-detailed','MaxIterations',50);
%options = optimset('Display','iter');
f = @(x)funcSinkhornSync(x,SynthData);
%[x,fval] = fminunc(f,X0(:,2:end),options);
[x,fval] = lsqnonlin(f,X0(:,2:end),[],[],options);
%[x,fval] = fminsearch(f,X0(:,2:end),options);

return ;

lambda = 2; 
I = SynthData.I;
e = length(I);
for k=1:e
    i = I(1, k); j = I(2, k);
    xi = sol{i};
    xj = sol{j};
    yi = solY{i};
    yj = solY{j};
    r = xi * (1.0./xj');
    x = r(:);
    y = yi*yj';
    y = y(:);
    y = y./sum(y);
    
    px = SynthData.Ratios{k};
    py = SynthData.RatiosY{k};
    
    M = pdist2(x(:),px(:),'cityblock');
    % M = M - diag(diag(M));
    
    a = y;
    b = py;
    K = exp(-lambda.*M);
    K(K<1e-100)=1e-100;
    U = K.*M ;
    [D,L,u,v]=sinkhornTransport(a,b,K,U,lambda,[],[],[],[],1);
    
    hold off; stem(x,y);
    hold on, stem(px,py);
    D
    pause;
end
