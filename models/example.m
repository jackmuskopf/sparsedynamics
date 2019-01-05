

sbioloadproject lotka m1
exportedModel = export(m1);
accelerate(exportedModel);

% save exportedLotka exportedModel

% To speed up compilation, we use the option |-N -p simbio|, which informs
% |mcc| that the deployed application does not depend on any additional
% toolboxes. For the purposes of this example, we programmatically
% construct the |mcc| command.
mccCommand = ['mcc -m simulateLotkaGUI.m -N -p simbio -a exportedLotka.mat ' ...
    sprintf(' -a %s', exportedModel.DependentFiles{:})];
% Uncomment the following line to execute the |mcc| command. This may take
% several minutes.
%
eval(mccCommand)
