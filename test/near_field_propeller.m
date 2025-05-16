clear all
close all
MM = [0 0.2 0.4 0.6 0.8 0.9];
table = readmatrix('example_prop.xlsx');
% Constants
n = 101;
R = 5; %ft
D = 2*R;
x = -0.5*D:(D/(n-1)):0.5*D;
p_rms = zeros(length(x), length(MM));

% viridis_colors = [0.2670, 0.0049, 0.3294;
%                   0.2810, 0.0896, 0.4125;
%                   0.2539, 0.2652, 0.5293;
%                   0.1638, 0.4712, 0.5581;
%                   0.1341, 0.6581, 0.5172;
%                   0.4775, 0.8219, 0.3181];
colors = jet(length(MM));

for K = 1:length(MM)
    M = MM(K);
    T = interp1(table(:,1),table(:,6),M);
    Q = 2680; %ft-lb
    B = 2;
   
    Re = 0.8*R;
    m = 1;
    k = 0.29686;
    beta = sqrt(1-M^2);
    
    y = 0.6*D;
    x1 = 0;
    z = 0;
    
    %S = sqrt((x-x1).^2+beta.^2*(y-y1).^2);
    theta = 0:2*pi/(n-1):(2*pi-2*pi/(n-1));
    AA = zeros(1,length(x));
    BB = zeros(1,length(x));
    S = zeros(1,length(x));
    
    
    for i = 1:length(x)
        % S(i) = sqrt((x(i)-x1).^2+beta.^2*(y-y1).^2);
        for j = 1:length(theta)
            %S = sqrt((x(i)-x1).^2+beta.^2*(y-y1).^2);
            y1 = Re*cos(theta(j));
            z1 = Re*sin(theta(j));
            S(i) = sqrt((x(i)-x1)^2 + beta^2*((y-y1)^2 + (z-z1)^2));
            sigma = (M*(x(i)-x1) + S(i))/beta^2;
            a1 = T*x(i)/S(i)^2*cos(m*B*theta(j)+k*sigma);
            a2 = (T*k/beta^2*(M+x(i)/S(i))-Q*m*B/Re^2)*sin(m*B*theta(j)+k*sigma);
            AAj = (a1+a2)/S(i);
            b1 = -T*x(i)/S(i)^2*sin(m*B*theta(j)+k*sigma);
            b2 = (T*k/beta^2*(M+x(i)/S(i))-Q*m*B/Re^2)*cos(m*B*theta(j)+k*sigma);
            BBj = (b1+b2)/S(i);
            AA(i) = AAj+ AA(i);
            BB(i) = BBj +BB(i);
        end
        p_rms(i, K) = sqrt(2)/(8*pi^2)*sqrt((AA(i)*478.8)^2+(BB(i)*478.8)^2);
    end
    plot(x/D,p_rms(:, K), 'DisplayName',strcat(['M = ', num2str(M)]), ...
        'Color',colors(K, :))
    hold on
end
grid on
grid minor
legend()
%ylim([0 2200])