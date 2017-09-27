%% ********************************************************
%*Description: HSIC Lasso for solving house price feature selection
%*Data: Kaggle house pricing data (only for assignment usage) (distribution could cause legal issues, please don't)
%*Author: Chen Wang, UCL Dept. of Computer Science
%*Reference: Yamada et al. 2013, High-Dimensional Feature Selection by Feature-Wise Kernelized Lasso
%*Reference: Yamada et al. related software
%***********************************************************
clc;
clear;
close all
%% load the data
data_path = '../data/';
%load DAL package
addpath(genpath('DAL_opt'));
% addpath(genpath('zeroSR1_opt'));
% run setup_zeroSR1.m;
%Load 
%Train Data
% Train_Data = csvread(strcat(data_path,'train_data.csv'),1,1);
% TrainX = Train_Data(:,1:end-1);
% TrainY = Train_Data(:,end);
%Normalized Training Data
%load the data from .mat document
load TrainX TrainX
load TrainY TrainY
TrainX_Normalized = TrainX./repmat(max(TrainX,[],1),size(TrainX,1),1);
%Test Data
% Test_Data = csvread(strcat(data_path,'test_data.csv'),1,1);
% TestX = Test_Data(:,1:end-1);
% TestY = Test_Data(:,end);
%load the mat data
load TestX TestX
load TestY TestY
%Normalized Test Data
TestX_Normalized = TestX./repmat(max(TestX,[],1),size(TestX,1),1);
%fix the random parameter
rand('seed',2018);
%get 5 random data for d>n^2 test
index_5 = randi([1,size(TrainX,1)],5,1);
Sparse_data_train_X = TrainX(index_5,:);
Sparse_data_train_Y = TrainY(index_5,:);
%get 50 random data for d<n^2 test
index_50 = randi([1,size(TrainX,1)],50,1);
Sparse_dim_train_X = TrainX(index_50,:);
Sparse_dim_train_Y = TrainY(index_50,:);

%% A feature dictionary to Illustrate if our method works well
FeatureDic = {'MSSubClass','MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour',...
    'Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',...
    'OverallQual','OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd',...
    'MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',...
    'BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical',...
    '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',...
    'KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt',...
    'GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch',...
    '3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition'...
};

%% Main Routine to call HSIC with different method
%% Part I: Proximal Gradient methods
lambda = 1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'proximal_gradient');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for proximal gradient method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'proximal_gradient');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for proximal gradient method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part II: Proximal and Accelerated Proximal Gradient methods
fprintf('\n\n');
lambda = 0.1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'acc_proximal_gradient');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for accelerated proximal gradient method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'acc_proximal_gradient');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for accelerated proximal gradient method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part III: FISTA with constant
fprintf('\n\n');
lambda = 1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'FISTA_const');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for constant-step FISTA method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'FISTA_const');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for constant-step FISTA method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part IV: FISTA with backtrack method
fprintf('\n\n');
lambda = 1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'FISTA_backtrack');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for backtrack-step FISTA method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'FISTA_backtrack');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for backtrack-step FISTA method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part V: Projected Newton with constant step
fprintf('\n\n');
lambda = 0.5;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'Newton_proximal_const');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for constant-step Newton Proximal method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'Newton_proximal_const');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for constant-step Newton Proximal method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part VI: Projected Newton with backtracking step
fprintf('\n\n');
lambda = 0.1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'Newton_proximal_backtrack');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for backtracking-step Newton Proximal method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'Newton_proximal_backtrack');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for backtracking-step Newton Proximal method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part VII: Dual Augmented Lagrangian (Implement)
fprintf('\n\n');
lambda = 0.1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'DAL');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for Dual Augmented Lagrangian method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'DAL');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for Dual Augmented Lagrangian method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part VIII: ADMM method
fprintf('\n\n');
lambda = 0.1;
[alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'ADMM');
[~,slected_feature] = sort(alpha_n,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for ADMM method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
[alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'ADMM');
[~,slected_feature] = sort(alpha_m,'descend');
most_features = feature_filter(slected_feature);
current_selected_feature = FeatureDic(most_features);
fprintf('There are %d non-zero alpha value for ADMM method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
fprintf('The selected most notable features are:');
disp(current_selected_feature);
fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);

%% Part extra: DAL method with package
%****************************************
%This is the DAL method with integrated package, which could somehow be
%used as a 'benchmark' criteria to see if the algorithms works well
%In the original proposal of the HSIC Lasso paper, the author suggested to
%use this. This is also the original software provided by the HSIC Lasso
%authors.
%****************************************
% fprintf('\n\n');
% lambda = 0.1;
% [alpha_n,optvalue_n,info_n] = HSIC_feature_selection(Sparse_data_train_X,Sparse_data_train_Y,lambda,'DAL_package');
% [~,slected_feature] = sort(alpha_n,'descend');
% most_features = feature_filter(slected_feature);
% current_selected_feature = FeatureDic(most_features);
% fprintf('There are %d non-zero alpha value for DAL (package) method with data-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_n>0),1),optvalue_n);
% fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_n.steps,info_n.time);
% [alpha_m,optvalue_m,info_m] = HSIC_feature_selection(Sparse_dim_train_X,Sparse_dim_train_Y,lambda,'DAL_package');
% [~,slected_feature] = sort(alpha_m,'descend');
% most_features = feature_filter(slected_feature);
% current_selected_feature = FeatureDic(most_features);
% fprintf('There are %d non-zero alpha value for DAL (package)  method with dim-sparse mode and the nomrlized objective function value is:%4.3f\n',size(find(alpha_m>0),1),optvalue_m);
% fprintf('The algorithm takes %d steps to converge and the time cost is: %4.4f\n\n',info_m.steps,info_m.time);