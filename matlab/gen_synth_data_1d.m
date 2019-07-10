function [SynthData] = gen_synth_data_1d(N, completeness, maxNumModes, numParticles)

%% generate the pose graph edges
Npair = N*(N-1)/2;
I = zeros(2, Npair);

% generate all pairwise edges
k=1;
for i=1:N
    for j=i+1:N
        if(i~=j)
            I(:,k)=[i; j]; k=k+1;
        end
    end
end

% now keep a portion of the edges
e = ceil(completeness*Npair);
ind = randperm(Npair, e);
I = I(:, ind);
vals = ones(1, e);
G = sparse(I(1,:), I(2,:), vals, N, N);
e = length(I);

Xs = cell(N, 1);
Ys = cell(N, 1);
YsWeight = cell(N, 1);
Ratios = cell(e, 1);
RatiosY = cell(e, 1);
RatiosYWeight = cell(e, 1);
DistrGndTruth = cell(N, 1);

% generate emprical prior distributions for each node
for i=1:N
    
    numModes = randi(maxNumModes);
    
    mus = randn(numModes, 1);
    vars = rand(numModes, 1)*0.3;

    [X, gmx] = UMGRN(mus,vars,numParticles,'with_plot', 0 );
    Y = gmx(X);
    Y = Y./sum(Y); % simplex
    
    Xs{i} = X';
    Ys{i} = ones(length(X), 1);
    YsWeight{i} = Y';
    DistrGndTruth{i} = [mus; vars]; 
end

for k=1:e
    i = I(1, k); j = I(2, k);
    xi = Xs{i};
    xj = Xs{j};
    yi = Ys{i};
    yj = Ys{j};
    ywi = YsWeight{i};
    ywj = YsWeight{j};
    r = xi * (1.0./xj');
    r = r(:);
    y = yi*yj';
    yw = ywi*ywj';
    y = y(:);
    yw = yw(:);
    y = y./sum(y);
    yw = yw./sum(yw);
    Ratios{k} = r;
    RatiosY{k} = y;
    RatiosYWeight{k} = yw;
end

% now obtain the pairwise

SynthData.I = I;
SynthData.G = G;
SynthData.Xs = Xs;
SynthData.Ys = Ys;
SynthData.YsWeight = YsWeight;
SynthData.DistrGndTruth = DistrGndTruth;
SynthData.Ratios = Ratios;
SynthData.RatiosY = RatiosY;
SynthData.RatiosYWeight = RatiosYWeight;
SynthData.completeness = completeness;
SynthData.N = N;
SynthData.numParticles = numParticles;

end

