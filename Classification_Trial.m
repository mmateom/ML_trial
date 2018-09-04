%% Classification trial

clearvars;
% close all;
clc;
set(0,'defaultfigurewindowstyle','docked');


%% Load data in form of atable
data = load ('dataYoung_filt_table.mat');
data = data.data_table;

%% New approach - Create training, dev and test data
% Training data set will be splitted in Training and Dev
% to perform cross-validation with LOSO - leave-one-subject-out

% c = cvpartition(data.Activity,'HoldOut',0.3);
% idxTrain = training(c);
% dataTrain = data(idxTrain,:);
% idxTest = test(c);
% dataTest = data(idxTest,:);

%% Old approach - create training and test data
% directly perform CV. The training and test data
% are created during CV. Mannini used old approach on his papers.

%% Create models

%BEST PARAMS FOR mSVM
%     Coding     BoxConstraint    KernelScale
%    ________    _____________    ___________

%    onevsone       935.78          11.664 

% Try different models...
% mdlKnn = fitcknn(dataTrain,'Activity','CrossVal','on');
% mdlDa = fitcdiscr(dataTrain,'Activity', 'CrossVal','on');

%in templateSVM, default is one-versus-one
template = templateSVM('KernelFunction','gaussian','BoxConstraint',935,'KernelScale',11);
mdlSVM = fitcecoc(data,'Activity','Learners',template); %'CrossVal','on'
%mdlSVM = fitcecoc(data{:,1:end-1},data.Activity,'OptimizeHyperparameters','auto','Learners','linear');

%OptimizeHyperparameters - default method: bayesian. Can perform gradient
%descent and grid search

%% Evaluate model performance

%train error
train_error = resubLoss(mdlSVM);

%cross validate
CVMdl = crossval(mdlSVM);%cross-validate SVM. Default - 10-fold

%test test error
lossSVM = kfoldLoss(CVMdl);%test error

% preKnn = predict(mdlKnn,dataTest);
% preDa = predict(mdlDa,dataTest);
% preSVM = predict(mdlSVM,dataTest);
%%use kfoldPredict when kfold, crossval or loso are 'on'
% preKnn = kfoldPredict(mdlKnn);
% preDa = kfoldPredict(mdlDa);

%Confusion Matrix
[pred, scores] =  kfoldPredict(CVMdl);
[cm, grp] =confusionmat(data.Activity,pred);
stats = confusionmatStats(cm);%custom function from community
accuracy = stats.accuracy*100;%make it a percentage

%% Create a table to hold the results
% modelNames = {'kNN','Discriminant Analysis','SVM'};
% results = table([lossKnn;lossDa;lossSVM],'RowNames',modelNames,...
%     'VariableNames',{'kFoldLoss'});

%% Display the results

figure(2);
%plotconfusion(data{:,end}',validationPredictions')%always row vectors
heatmap(grp,grp,cm);
title('Confusion Matrix');
set(gca,'FontSize',18) 
colormap summer

figure(3);
x = categorical({'train','test'});
x = reordercats(x,{'train' 'test'});
y = [train_error,lossSVM];
b = bar(x,y);
b.FaceColor = 'flat';
b.CData(2,:) = [.5 0 .5];
set(gca,'FontSize',18) 
title('Train vs. Test Error')

figure(4);
x = categorical({'Ambulation','Cycling','Other','Sedentary'});
x = reordercats(x,{'Ambulation' 'Cycling' 'Other' 'Sedentary'});
y = accuracy';%now it's a row vector
b = bar(x,y);
set(gca,'FontSize',18) 
title('Accuracy')

