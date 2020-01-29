%% non-linearly separable classification - back propagation


sd = .85;

x1 = [normrnd(0,sd,50,1); normrnd(0,sd,50,1);  normrnd(0,sd,50,1)];
x2 = [normrnd(0,sd,50,1); normrnd(5,sd,50,1);   normrnd(10,sd,50,1)];
x3 = [ones(150,1)];
y1 = [ones(50,1) zeros(50,1) zeros(50,1);  zeros(50,1) ones(50,1) zeros(50,1); zeros(50,1) zeros(50,1) ones(50,1) ];
input = [x1 x2 x3];
output = y1;

nsamp = length(x1);
ninput = 3;
nhidden = 4;
noutput = 3;

W = unifrnd(-1,1,ninput,nhidden);  % initialize weight matrices
V = unifrnd(-1,1,nhidden,noutput);

mu = .05; p = .9;   % a suggested step and momentum size

lastdW = 0*W;  lastdV = 0*V;   % initialize the previous weight change variables

% now do back prop

%%  k-means clustering

x1 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];    % the data clusters
x2 = [normrnd(0,1,50,1); normrnd(5,1,50,1)];
input = [x1 x2];
nsamp = length(input);

w1 = mean(input) + normrnd(-1,1,1,2);   % intiialize the weights somewhere in the center of the data
w2 = mean(input) + normrnd(-1,1,1,2);


%%  ML density estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% mixture model using maximum likelihood gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1 = [normrnd(0,1,150,1); normrnd(7,2,150,1)];    % the data clusters
x2 = [normrnd(0,1,150,1); normrnd(7,2,150,1)];
input = [x1 x2];
nsamp = length(input);

w1 = mean(input) + normrnd(-1,1,1,2);            %initialize the means and standard deviations
w2 = mean(input) + normrnd(-1,1,1,2);
s1 = sqrt(mean(var(input)));
s2 = sqrt(mean(var(input)));
pi1 = .5;                                      % the prior probabilities - we're not fitting those here
pi2 = .5; 


%%  you can use the function (or just the code) below to make plots and track the current Gaussians

function plot_Gaussians(dat, w1,w2,s1,s2)
% plots the means and sd for isotropic Gaussians, superimposed on the data
% DAT.  Takes means w1 and w2, with standard deviations S1 and S2, and
% plots the mean and 2*SD for each plot
    
    clf                                                                     % plot the progress
    plot(x1,x2,'.')
    hold on
    plot(w1(1),w1(2),'r+')
    plot(w2(1),w2(2),'g+')
    rad1 = norminv(.75,0,2*s1);
    rad2 = norminv(.75,0,2*s2);
    theta = -pi:.2:pi;
    SDring= [cos(theta)' sin(theta)'];
    plot(rad1*SDring(:,1) + w1(1) ,rad1*SDring(:,2)+w1(2),'r')
    plot(rad2*SDring(:,1) + w2(1) ,rad2*SDring(:,2)+w2(2),'g')
    axis('equal')
    drawnow
    hold off

%% 1D Kohonen network
    
% create the data
x = -10:.1:10;  % equally spaced x
y =x.^3+ 0*x.^2;   % the cubic function
y = y/max(y)+ normrnd(0,.1,size(x));   % x = y^3 + noise; with y normalized to stay in range
input = [x; y]';  % this is the data set to fit your map to

nunits = 20;                    % the number of units

% initialize the prototypes to be in the center of the data
W = [mean(x)+unifrnd(-.05,.05,nunits,1) mean(y)+unifrnd(-.05,.05,nunits,1)];   

