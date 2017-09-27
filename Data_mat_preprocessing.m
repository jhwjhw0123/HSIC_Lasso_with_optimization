%Use this to process the .cvs document into .mat data
%IMPORTANT: The data is used for the assignment ONLY. Because it is
%real-life data collected from corporation, using it for other usage might
%cause legal issues
%% load the data
data_path = '../data/';
%load DAL package
% addpath(genpath('DAL_opt'));
% addpath(genpath('zeroSR1_opt'));
% run setup_zeroSR1.m;
%Load 
%Train Data
Train_Data = csvread(strcat(data_path,'train_data.csv'),1,1);
TrainX = Train_Data(:,1:end-1);
TrainY = Train_Data(:,end);
save TrainX TrainX
save TrainY TrainY
%Test Data
Test_Data = csvread(strcat(data_path,'test_data.csv'),1,1);
TestX = Test_Data(:,1:end-1);
TestY = Test_Data(:,end);
save TestX TestX
save TestY TestY
