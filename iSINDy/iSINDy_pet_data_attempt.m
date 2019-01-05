%% implicit SINDy attempt

% David O'Gara
% WUSTL ESE
% 12/11/2018

%prepare workspace
clear all
close all
clc
addpath('./utils');
%read in data:

data = readtable('measurement_data.csv');
dxt = [zeros(1,size(data{:,:},2));diff(data{:,:})];

% one variable for now
t = data.Var1;
x = data.x4;

%Easy way to access data:
%     t = data.Var1;
%     x1 = data.x1;
% 
%     plot(t,x1)

%pool Data

% search for equation for the 1st state-variable
% We do the ADM method for only one value of lambda 
% see 2nd state varaiable for full sweep.
% clear results for other state variables
clear Theta Thetastring Xi indTheta lambdavec numterms errorv indopt
clear indTheta1 Xi1 numterms1 nT

% pool Data  (i.e., build library of nonlinear time series)
%xt = data{:,:};
%[num_t, n] = size(data{:,:});


%% set the functions to search
% may not be the same as the functions we used to generate the data
laurentorder = 0; % n, for 1/x^n terms in the library
polyorder = 3; % n, for polynomial terms in the library x^n
usesine = 0; % adds sine to the library
dyorder = 1; % n for (dx/dt)^n terms in the library
n = 1;
xt = x;
x0=xt(1);
% compute TVRegDiff for dx
dxt = [0;diff(xt)];


% pool Data  (i.e., build library of nonlinear time series)
[Theta, Thetastring] = poolDatady(xt,n,polyorder,usesine, laurentorder, dxt, dyorder);

%% compute Sparse regression using ADM
pflag = 2; %plot output. 1 will plot some pareto front output, 2 will plot ADM algorithm details.
tol = 1e-3; %updated tol
[Xi, indTheta, lambdavec, numterms, errorv] = ADMpareto(Theta, tol, pflag);

%% compare infered model and original model for new initial conditions


%% numerically simulate time series for each SINDy discovered model to validate
[libsize, nummods] = size(Xi); % find the number of models
x1 = x;
t1 = data.Var1;
for kk = 1:nummods % loop through sparse coefficient vectors
    kk; % display the set of coefficients we are on
    
    tspan=data.Var1;
    for i = 1:length(x0)
        %newdxdt(x)
        figure('Name',['Model',num2str(kk)])
        subplot(1,2,2)
        hold on
        [t2,x2] = ode45(@(t,x)newdxdt(x,Xi,kk), tspan, x0(i));
        title('Simulated Model')
        plot(t2,x2)
        subplot(1,2,1)
        plot(t1,x1)
        title('True Model')
        
    end
    xlabel('time')
    ylabel('x')
    drawnow
    hold off
    
    % save validation time series
    x2val{kk} = x2;
    t2val{kk} = t2;
    
    % store Root Mean Square error between data and validation time series
    %RMSE(kk) = sqrt(mean((x2-x1).^2));
    
end
% Plot Pareto 
figure
semilogy(numterms, errorv, 'o')
xlabel('Number of terms')
ylabel('Validated RMSE')
title('Pareto Front for Validation')


% Plot Pareto 
figure
plot(numterms, errorv, 'o')
xlabel('Number of terms')
ylabel('Validated RMSE')
title('Pareto Front for Validation')


function dx = newdxdt(x,Xi,kk)

num = Xi(1:size(Xi,1)/2, kk)'*[1 x x*x x*x*x]';
denom = (Xi(size(Xi,1)/2+1:end, kk)'*[1 x x*x x*x*x]');


dx = num/denom;

end
