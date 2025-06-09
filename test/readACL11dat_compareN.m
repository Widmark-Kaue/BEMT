% addpath('/home/kleine/Dropbox/KTH/Routines/Matlab/mat2tikz/src') 
Ndata=1;
plotonlyforces=1;

%% Parameters in Nek for writing file
period=2*pi/7.55;
Tmax=10;
%Tmax=1.5;
nt=20;
indexTmax=Tmax*nt+1;
% nacl=42;
nacl=22;

%% Just for plotting
markernek(1,:) = '--o';
markernek(2,:) = 'd  ';
markernek(3,:) = '--s';
markernek(4,:) = '+  ';
markernek(5,:) = '--d';
markernek(6,:) = 'x  ';
markersize(1) = 6;
markersize(2) = 8;
markersize(3) = 6;
markersize(4) = 8;
markersize(5) = 4;
markersize(6) = 8;
colornek=zeros(Ndata,3);
colornek(1,:)=[0 0.5 0];
colornek(2,:)=[0.5 1 0];
colornek(3,:)=[0 0 1];
colornek(4,:)=[0 1 1];
colornek(5,:)=[1 0 0];
colornek(6,:)=[1 0.5 0];

%% Read files
% (time,pts_ACL,V_s,V_t,V_n,x,y,z,circ,Fn,Ft)
%ACLfilename='ACL11_dire_noshear.dat';
% ACLfilename='ACL11_dire35_aftrev.dat';
ACLfilename='ACL11.dat';
datanekfull=load(ACLfilename);
datanekfull2 = reshape(datanekfull,nacl,[],11);
datanekfull2 = permute(datanekfull2,[2 1 3]);
datanek=zeros([size(datanekfull2(1:indexTmax,:,:)) Ndata]);
datanek(1:indexTmax,:,:,1)=datanekfull2(1:indexTmax,:,:);

% ACLfilename='ACL11_dire7_aftrev.dat';
% datanekfull=load(ACLfilename);
% datanekfull2 = reshape(datanekfull,nacl,[],11);
% datanekfull2 = permute(datanekfull2,[2 1 3]);
% datanek(1:indexTmax,:,:,2)=datanekfull2(1:indexTmax,:,:);
% 
% ACLfilename='ACL11_dire35_aftrev.dat';
% datanekfull=load(ACLfilename);
% datanekfull2 = reshape(datanekfull,nacl1,[],11);
% datanekfull2 = permute(datanekfull2,[2 1 3]);
% datanek(1:indexTmax,:,:,3)=datanekfull2(1:indexTmax,:,:);
% 
% ACLfilename='ACL11_dire35_aftrev.dat';
% datanekfull=load(ACLfilename);
% datanekfull=load(ACLfilename);
% datanekfull2 = reshape(datanekfull,nacl1,[],11);
% datanekfull2 = permute(datanekfull2,[2 1 3]);
% datanek(1:indexTmax,:,:,4)=datanekfull2(1:indexTmax,:,:);
% 
% ACLfilename='ACL11_dire7_aftrev.dat';
% datanekfull=load(ACLfilename);
% datanekfull2 = reshape(datanekfull,nacl1,[],11);
% datanekfull2 = permute(datanekfull2,[2 1 3]);
% datanek(1:indexTmax,:,:,5)=datanekfull2(1:indexTmax,:,:);
% 
% ACLfilename='ACL11_dire7_aftrev.dat';
% datanekfull=load(ACLfilename);
% datanekfull=load(ACLfilename);
% datanekfull2 = reshape(datanekfull,nacl1,[],11);
% datanekfull2 = permute(datanekfull2,[2 1 3]);
% datanek(1:indexTmax,:,:,6)=datanekfull2(1:indexTmax,:,:);

% %% Read DTU files (for validation)
% DTUfl=load('Fl_DTU.dat');
% DTUfd=load('Fd_DTU.dat');

%% Calculate other parameters
%phi (flow angle)
datanek(:,:,12,:)=atan2(datanek(:,:,5,:),-datanek(:,:,4,:));
%lift
datanek(:,:,13,:)=datanek(:,:,10,:).*cos(datanek(:,:,12,:))+datanek(:,:,11,:).*sin(datanek(:,:,12,:));
%drag
datanek(:,:,14,:)=datanek(:,:,10,:).*sin(datanek(:,:,12,:))-datanek(:,:,11,:).*cos(datanek(:,:,12,:));

%% Plot figures
if plotonlyforces==0

figure(1)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(1:indexTmax,nacl-1,1,idata)/period,datanek(1:indexTmax,nacl-1,9,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
%xlim([0 12])
xlabel('$t/T$','Interpreter','latex')
ylabel('$\Gamma$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on
%matlab2tikz('6Gamma_t_point.tex');

figure(3)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,9,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Gamma$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on
%matlab2tikz('6Gamma_r.tex');

figure(13)
hold on
set(gcf,'color','w');
for idata=2:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,9,idata)-datanek(indexTmax,2:nacl-1,9,1),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Delta \Gamma$','Interpreter','latex')
legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','Interpreter','latex')
grid on
%%matlab2tikz('DGamma_rxx.tex');

figure(23)
hold on
set(gcf,'color','w');
for idata=2:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),(datanek(indexTmax,2:nacl-1,9,idata)-datanek(indexTmax,2:nacl-1,9,1))/max(abs(datanek(indexTmax,2:nacl-1,9,1))),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Delta \Gamma/max(\Gamma)$','Interpreter','latex')
%legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','Interpreter','latex')
legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 2 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 2 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 2 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 2 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 2 \Delta x$))','Interpreter','latex')
grid on
%matlab2tikz('6DGammanorm_r.tex');

end

figure(4)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,13,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
%plot(DTUfl(:,1),DTUfl(:,2)*2,'*')
xlabel('$r$','Interpreter','latex')
ylabel('$f_l$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on
%matlab2tikz('6fl_r.tex');

figure(5)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,14,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
%plot(DTUfd(:,1),DTUfd(:,2)*2,'*')
xlabel('$r$','Interpreter','latex')
ylabel('$f_d$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on
%matlab2tikz('6fl_d.tex');

if plotonlyforces==0

figure(6)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,12,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\phi$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on

figure(14)
hold on
set(gcf,'color','w');
for idata=2:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,13,idata)-datanek(indexTmax,2:nacl-1,13,1),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Delta f_l$','Interpreter','latex')
%legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','Interpreter','latex')
grid on

figure(15)
hold on
set(gcf,'color','w');
for idata=2:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,14,idata)-datanek(indexTmax,2:nacl-1,14,1),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Delta f_d$','Interpreter','latex')
%legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','Interpreter','latex')
grid on

figure(17)
hold on
set(gcf,'color','w');
for idata=2:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,4,idata)-datanek(indexTmax,2:nacl-1,4,1),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Delta u_t$','Interpreter','latex')
legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','Interpreter','latex')
grid on
%%%matlab2tikz('Du_rxx.tex');

figure(18)
hold on
set(gcf,'color','w');
for idata=2:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,5,idata)-datanek(indexTmax,2:nacl-1,5,1),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$\Delta u_n$','Interpreter','latex')
legend('$\Gamma$(ALM$^i$ ($\varepsilon = 3.5 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^d$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','$\Gamma$(ALM$^i$ ($\varepsilon = 7 \Delta x$)) - $\Gamma$(ALM$^d$ ($\varepsilon = 3.5 \Delta x$))','Interpreter','latex')
grid on
%%%matlab2tikz('Du_rxx.tex');

end

figure(8)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,10,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$f_n$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on
%matlab2tikz('6fl_r.tex');

figure(9)
hold on
set(gcf,'color','w');
for idata=1:Ndata
  plot(datanek(indexTmax,2:nacl-1,2,idata),datanek(indexTmax,2:nacl-1,11,idata),markernek(idata,:),'MarkerSize',markersize(idata),'Color',colornek(idata,:),'LineWidth',1.0)
end
xlabel('$r$','Interpreter','latex')
ylabel('$f_t$','Interpreter','latex')
legend('ALM$^d$ ($\varepsilon = 3.5 \Delta x$)','ALM$^i$ ($\varepsilon = 3.5 \Delta x$)','ALM$^d$ ($\varepsilon = 7 \Delta x$)','ALM$^i$ ($\varepsilon = 7 \Delta x$)','Interpreter','latex')
grid on
%matlab2tikz('6fl_d.tex');
