function [H_f,Ws_f,Hs_f,obj_vals,n_iter,loc] = dNMF(X,v,alpha,beta,k);

%%
% 
% H ：k*n
% Wt{i,1}： n*k
% Ht{i,1}： k*n
% k： dimension

[cc,dd] = size(X);
if(cc ~= length(X))
    X = X';
end

[m,n] = size(X{1,1});    % m表示行，n表示列
H_f = zeros(k,n);

for i = 1:v  
    %最近邻
    %A0 = constructW_PKN(p{i,1}',10, 0);
    %A1 = (A0+A0')/2;
    A{i,1} = X{i};
    Di = diag(sum(A{i,1}));
    D{i,1} = Di;   
end
for i = 1:v
    %X{i} = PMI(X{i},3);
    Hs_f{i,1} = zeros(k,n);
    Ws_f{i,1} = zeros(m,k);
end

obj_vals = [];
%%Update
%for d = 1:20
    new_eval = 0;
    count = 0;

    %%   使用SVD进行初始化
    H = rand(k,n);  %common
    for i = 1:v
        %[U,S,V] = svds(X{i,1},10);
        [U,S,V]=svds(X{i,1},k);
        U1=abs(U*S^0.5);
        V1=abs(S^0.5*V');
        W{i,1} = U1;
        Ht{i,1} = V1; 
    end

    %%
    H_f = H;
    Hs_f = Ht;
    Ws_f = W;
    
    %% 更新
    while(count<=100)%&& abs(old_eval - new_eval)>(start_eval - new_eval)*(1e-2))
       
        
        for i = 1:v
            
            XH = {};
            WH = {};
           %% update Ht 
        
            WX{i,1} = 2*W{i,1}'*X{i,1}+ 2*beta*Ht{i,1}*A{i,1};
            H_sum = zeros(k,n);
            for j = 1:v
                if(i~=j)
                    H_sum  = H_sum + Ht{j,1};
                end
            end

            H_sum = H_sum./(v-1);
            WH2{i,1} = 2*W{i,1}'*W{i,1}*(Ht{i,1}+H) + alpha*H_sum + 2*beta* Ht{i,1} *D{i,1};
            Ht{i,1} = Ht{i,1}.*(WX{i,1}./max(WH2{i,1},1e-10));
            N = normalize(Ht{i,1},'norm',2);
            Ht{i,1} = N; 
               %% update W
            XH{i,1} = X{i,1}*(H+ Ht{i,1})';
            WH{i,1} = W{i,1}*(H+ Ht{i,1})*(H+ Ht{i,1})';
            W{i,1} = W{i,1}.*(XH{i,1}./max(WH{i,1},1e-10));
            N = normalize(W{i,1}','norm',2);
            W{i,1} = N';        
           
            %% update H 
             XT = zeros(k,n);
             WHH = zeros(k,n);
             for ii = 1:v
                 XT = XT + W{ii,1}'*X{ii,1};   %d*n
                 WHH = WHH + W{ii,1}'*W{ii,1}*(H + Ht{ii,1});
             end
             H = H.*(XT./max(WHH,1e-10));
             N = normalize(H,'norm',2);
             H = N; 
        end
     

        if count == 0
            new_eval = NMF_obj(X,H,Ht,W,A,D,alpha,beta,v);
        end

        if (count~=0) %&& mod(count,100)==0)
            old_eval = new_eval;
            [new_eval,new_ev] = NMF_obj(X,H,Ht,W,A,D,alpha,beta,v);
            if old_eval > new_eval
                H_f = H;
                Hs_f = Ht;
                Ws_f = W;
            end
            obj_vals(1,count) = log(new_eval);
            obj_vals(2,count) = log(new_ev);
            %fprintf('count= %d,new_eval is %.6f,\n',count,new_eval);
        end
        count = count + 1;
       % fprintf("====count : %d =====\n",count);
%         if (mod(count,10)==0)
%             count
%         end 
    end
    n_iter = count;
   
    %idx = kmeans(Hs_f,k)
    for i = 1:v
        %loci = kmeans(Hs_f{i}',k);
        [~, loci] = max(Hs_f{i,1},[],1);
        loc(:,i) = loci;
    end
    
%     fprintf('d= %d,new_eval is %.6f,',d,new_eval);
%end

%% 
function [obj_d,obj1] = NMF_obj(X,H,Ht,W,A,D,alpha,beta,v);

obj = 0;
pen = 0;
tr1 = 0;
obj1=0;
[k,m] = size(H);
ob = zeros(m,m);
for i = 1:v
    ob = abs(X{i,1} - W{i,1}*(Ht{i,1}+H));
    %ob = abs(X{i,1} - W{i,1}*(Ht{i,1}+H));
    
    obj1 = obj1 + norm(ob,'fro')/norm(X{i,1},'fro');
    obj = obj + norm(ob,'fro');
    
    %pen = pen + alpha * norm(W*Ht{i,1}','fro');
    H_sum = 0;
    for j = 1:v
        if(i~=j)
            H_sum  = H_sum + Ht{j,1};
        end
    end
    %H_sum = H_sum./(v-1);
    pen = pen + alpha*(trace(Ht{i,1}*H_sum'));
    tr1 = tr1 + beta*trace(Ht{i,1} *(D{i,1}-A{i,1}) *Ht{i,1}');
end
obj_d = obj+pen+tr1;
