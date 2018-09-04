%% Create table

% load StanfordDataset2010_adult_Wrist_win12800_corrected_classes1to3_apr2013_90Hz_filt_o4f20_recalib_dec2013D.mat
load StanfordDataset2010_youth_Wrist_win12800_corrected_classes1to3_apr2013_90Hz_filt_o4f20_recalib_dec2013D.mat

n = size(Data,2);       % number of subjects
f = size(Data(1).Pm3,1);% number of features

alldata = [];

%create table with features and labels
for i = 1:n

data_n = Data(i).Pm3';
labels = Labels(i).Lab5w;

alldata = [alldata; data_n labels];

end

%% Preprocessing - If needed
%Remove activities 0 and 1, we are not using them
idxZeros = find(alldata(:,end)==0);
alldata(idxZeros,:) =[];
idxOnes = find(alldata(:,end)==1);
alldata(idxOnes,:) =[];

%% Names of predictors and outcome
% names = {'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17',...
%     'F18','F19','F20','F21','F22','F23','F24','Activity'};
% 
%% Create table
% data_table = array2table(alldata,'VariableNames',names);
% 
% 
