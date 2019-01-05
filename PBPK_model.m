%% Sparse Dynamics for PBPK Model

% prepare workspace
clear all
close all
figpath = '../figures/';
addpath('./utils');
% set parameters
n = 10;
polyorder = 3;
usesine = 0;

% load data
data = load('REFSIM-25hr.mat');



x = 1000*data.x;  % 130t by 17 - state double



%clip to 130 by 10
x = x(:,1:10);
x0 = x(1,:)'; %get IC

%% compute Derivative
dx = [zeros(1,size(x,2)); diff(x)];

%% pool Data  (i.e., build library of nonlinear time series)
Theta = poolData(x,n,polyorder,usesine);

%add noise to Theta to preserve rank

Theta = Theta + 0.01*randn(size(Theta));

m = size(Theta,2);


% set parameters


%% compute Sparse regression: sequential least squares
lambda = 0.0000001;      % lambda is our sparsification knob.
Xi = sparsifyDynamics(Theta,dx,lambda,n);
state_names ={'VArt.CArt','VGut.CGut' ,'VGutLumen.AGutlumen', 'VLung.CLung','VVen.CVen', 'VRest.CRest' ,'VLiver.CLiver' , 'VLiver.CMetabolized' ,'VKidney.CKidney' ,'VKidneyTubules.CTubules'};
   %{
    {'QCardiac'               }
    {'QGut'                   }
    {'QLiver'                 }
    {'QKidney'                }
    {'QRest'                  }
    {'Qgfr'                   }
    {'VTotal'                 }
   %}

%initiliaze initial conditions
%x0 = [zeros(10,1)];
%x0(3,1) = 0.0093; % 'VGutLumen.AGutlumen'

newout = poolDataLIST(state_names,Xi,n,polyorder,usesine);

tspan = [0 25];
options = odeset('RelTol',1e-8,'AbsTol',1e-8*ones(1,n));
[tB,xB]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);  % approximate


% plot ODEs

for i = 1:10
   
    figure('Name',state_names{i});
    plot(tB,xB(:,i)');
    title(state_names{i});
    
end

% save plots
FolderName = 'C:\Users\david.ogara\Dropbox\ESE585\Sparse_dynamics\System Identification\sparsedynamics\figures\';   % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  FigName   = num2str(get(FigHandle, 'Number'));
  set(0, 'CurrentFigure', FigHandle);
  %saveas(fullfile(FolderName, [FigName '.jpg']));
  saveas(FigHandle,FigName,'jpg');
  
end
