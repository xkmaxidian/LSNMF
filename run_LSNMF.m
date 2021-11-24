clc;  close all; clear all;
currentFolder = pwd;
addpath(genpath(currentFolder));
resultdir = 'Results/';
if(~exist('Results','file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

dataname = {'syn_16'};
runtimes = 1; % run-times on each dataset, default: 1
numdata = length(dataname);
tic
for cdata = 1:numdata
    %% read dataset
    idata = cdata;
    %datadir = 'Datas/';
    dataf = [cell2mat(dataname(idata))];
    load(dataf);
     %X = X';
    %y0 = truelabel{1};
%  for i=1:length(X)
%     l=zeros(10000,1);
%     a=y{i,1}(:,1);
%     l(a)=y{i}(:,2);
%     Y(:,i)=l;
% end
    %% iteration ...
    for rtimes = 1:runtimes
        for i = 1:1
            v = length(X);
            for ii = 1:v
                y0 = Y(:,ii);
                k(ii) = length(unique(y0));
                k = max(k,[],2);
            end
            alpha = 0.5;  beta = 0.1;
            [H_f,Ws_f,Hs_f,obj_vals,n_iter,y1] = LSNMF(X,v,alpha,beta,k);
            for iii = 1:v
                 metric(iii,:) = ClusteringMeasure_new(Y(:,iii), y1(:,iii));
            end
        end;
        disp(char(dataname(idata)));
        result(cdata,:) = mean(metric);
        fprintf('=====In iteration %d=====\nACC:%.4f\tNMI:%.4f\tPu:%.4f\tF:%.4f\tARI:%.4f\n',rtimes,result(cdata,1),result(cdata,2),result(cdata,3),result(cdata,4));
        %save([resultdir,char(dataname(idata)),'_result.mat'],'result','alpha','beta');
        %clear ACC NMI ARI U y0 y1;
    end
end
toc
