clear all, close all, clc
figpath = '../figures/';
addpath('./utils');
addpath('./bioutils');

%% load Data
load('REFSIM-25hr.mat')

% reduce to the first dimension
x = x(:, 1:10);
names = names(1:10);

% size of system
%n = length(names);
n = 1;
tspan = t; % time vector
N = length(tspan);

dx = TVRegDiff(x(:, n), 100, 1);
dx = dx(2:end);
%dx = zeros(N, n);
%for ii = 1:N-1
%    dx(ii+1,:) = x(ii,1:n) - x(ii+1,1:n);
%end

dxt = dx;
xt = x;
options = odeset('RelTol',1e-7,'AbsTol',1e-7);

%% Plot Data
% Plot time series and derivatives of "training time series"
figure(5)
hold off
plot(t ,x, 'o')
xlabel('time')
ylabel('concentrations')
title('training time series')

%figure(6) 
%hold off
%plot(t, dx, 'o')
%xlabel('time')
%ylabel('derivative of concentrations w/ time')
%title('training derivative time series')

%% set the functions to search
% may not be the same as the functions we used to generate the data
laurentorder = 0; % n, for 1/x^n terms in the library
polyorder = 3; % n, for polynomial terms in the library x^n
usesine = 0; % adds sine to the library
dyorder = 1; % n for (dx/dt)^n terms in the library

% pool Data  (i.e., build library of nonlinear time series)
[Theta, Thetastring] = poolDatady(xt, n, polyorder, usesine, laurentorder, dxt, dyorder);

%% compute Sparse regression using ADM
pflag = 0; %plot output. 1 will plot some pareto front output, 2 will plot ADM algorithm details.
tol = 1e-5;
[Xi, indTheta, lambdavec, numterms, errorv] = ADMpareto(Theta, tol, pflag);

%% compare inferred model and original model for new initial conditions

figure(12)
x0 = x(1, :);
%dt = 0.1;
%tspan2=[0:dt:20];

%for i = 1:length(x0)
%    [t1,x1]=ode45(@(t,x)newdxdt(x(:,kk), Xi, kk),tspan2,x0(i),options);
%    x1 = x1 + eps*randn(size(x1));
%    plot(t1,x1, 'o')
%    hold on
%end

%% numerically simulate time series for each SINDy discovered model to validate
[libsize, nummods] = size(Xi); % find the number of models

for kk = 1:nummods % loop through sparse coefficient vectors
    kk % display the set of coefficients we are on

    xlabel('time')
    ylabel('x')
    tspan = [0:0.1:25];
    for i = 1:length(x0)
        [t2,y2]=ode23s(@(t,y)newdydt(y, Xi, kk), tspan, x0(i), options);
        y2
        plot(t2,y2)
        hold on
    end
    %drawnow
    %hold off
    
    % save validation time series
    %x2val{kk} = x2;
    %t2val{kk} = t2;
    
    % store Root Mean Square error between data and validation time series
    %RMSE(kk) = sqrt(mean((x2-x(length(x2),1)).^2));
end

% Plot Pareto with validation step
%figure
%semilogy(numterms, RMSE, 'o')
%xlabel('Number of terms')
%ylabel('Validated RMSE')
%title('Pareto Front for Validation')

function dy = newdydt(y, Xi, kk)
    %poly = [1 y y*y y*y*y y*y*y*y y*y*y*y*y]';
    poly = [1 y y*y y*y*y]';
    num = -Xi(1:size(Xi,1)/2, kk)'*poly;
    denom = (Xi(size(Xi,1)/2+1:end, kk)'*poly);
    dy = num/denom;
end
