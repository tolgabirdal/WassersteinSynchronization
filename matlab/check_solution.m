function [error] = check_solution(X, SynthData, initialSol)

a = ones(SynthData.numParticles,1);

error = 0;
figure,subplot(SynthData.N,1,1);
pi = 1;
for i=1:SynthData.N
    
    subplot(SynthData.N,1,i);
    hold on, stem(initialSol{i}, a);
    hold on, stem(X(:,i), 2*a);
    hold on, stem(SynthData.Xs{i}, 3*a);
    hold off;
    
    if (i==1)
        legend({'Initial','Optimized','Gnd Truth'});
    end
    
%     subplot(SynthData.N,3, pi+1);
%     stem(X(:,i), a);
%     if (i==1)
%         title('Optimized');
%     end
%     
%     subplot(SynthData.N,3, pi+2);
%     stem(SynthData.Xs{i}, a);
%     if (i==1)
%         title('Gnd Truth');
%     end
%     
    pi = pi+3;
    %pause;
    
    % error = error + norm(SynthData.Xs{i}-)
    
end



end
