% docs  https://www.mathworks.com/help/simbio/ref/sbiosimulate.html
% paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5026379/
% model https://www.ebi.ac.uk/biomodels-main/BIOMD0000000619
model = sbmlimport('data/BIOMD0000000619.xml');

% default config
config = getconfigset(model,'default');
config.TimeUnits = 'hour';
config.StopTime = 25;

% add a dose of Acetaminophen (APAP)
%dose1 = adddose(model,'d1','schedule');
%dose1.Amount = 1.4;
%dose1.AmountUnits = 'gram';
%dose1.TimeUnits = 'second';
%dose1.Time = 1;
%dose1.TargetName = 'VVen.CVen';

% run simulation
%[t,x,names] = sbiosimulate(model,config,dose1);
[t,x,names] = sbiosimulate(model,config);

% plot the amount of APAP in venous blood over time
figure(1);
venousAPAPInLiters = x(:, 6) / 3.41;
plot(t,venousAPAPInLiters);
xlabel('Time (hrs)');
ylabel('Venous APAP (Mol/L)');
title('Metabolism of Acetaminophen (APAP)');
% avoid exponential notation
ax = gca;
ax.YAxis.Exponent = 0;

compartments = [1,2,4,5,7,9,10];
figure(2);
plot(t,x(:, compartments));
xlabel('Time (hrs)');
ylabel('Compartment drug volume (Mol/L');
title('States vs Time');
legend(names(compartments));
ax = gca;
ax.YAxis.Exponent = 0;
