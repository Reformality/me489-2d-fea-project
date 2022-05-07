clc;
clear;

nodes = [ -0.50,         -0.50;
           0.00,         -0.50;
           0.50,         -0.50;
          -0.50,          0.00;
           0.00,          0.00;
           0.50,          0.00;
          -0.50,          0.50;
           0.00,          0.50;
           0.50,          0.50;
          -0.25,         -0.50;
           0.00,         -0.25;
          -0.25,          0.00;
          -0.50,         -0.25;
           0.25,         -0.50;
           0.50,         -0.25;
           0.25,          0.00;
           0.00,          0.25;
          -0.25,          0.50;
          -0.50,          0.25;
           0.50,          0.25;
           0.25,          0.50 ];

elements = [  1,  2,  5,  4, 10, 11, 12, 13;
              2,  3,  6,  5, 14, 15, 16, 11;
              4,  5,  8,  7, 12, 17, 18, 19;
              5,  6,  9,  8, 16, 20, 21, 17 ];
%-------------------------------------------------------------------------%
%                ME 489 Introduction to Finite Element Analysis           %
%                        Project function template                        %
%                                                                         %
% Note: 1) Use this function as a starting point for your code            %
%       2) Make sure all other functions that you need to run your code   %
%          are called from here. When the code is tested only this        %
%          function will be run.                                          %
%-------------------------------------------------------------------------%

% Problem 2:
NBC_flag = 1;
k = 1;
s = 2;
gamma_T = [1,4,8;0,0,0;1,4,8;0,0,0]; %local node number for Dirichlet BC (0s if this element not applicable)
T_bar = [0,-1,0,-1]; %temp for Dirichlet BC (-1 if this element not applicable)
J_det_NBC = 0.5/2; %NBC
q = -2; %NBC
% eta = 0; %temp value
% N_NBC = [0,-0.25*(1+1)*(1-eta)*(1-1+eta),-0.25*(1+1)*(1+eta)*(1-1-eta),0,0,0.5*(1+1)*(1-eta)*(1+eta),0,0];
NBC_elem = [0,1,0,1]; %enable value for element that has NBC
n_qp_NBC = 2;
qp_coord_NBC = [1,1/sqrt(3);1,-1/sqrt(3)];

%% program

%init
n_el = size(elements);
n_el = n_el(1); %number of elements
n_node = size(elements);
n_node = n_node(2); %number of nodes in one elements
n_qp = 4; %4-point quadrature
qp_coord = [-1/sqrt(3),-1/sqrt(3);1/sqrt(3),-1/sqrt(3);
    1/sqrt(3),1/sqrt(3);-1/sqrt(3),1/sqrt(3)];%quadrature point coordinates
C = 10^5;
K_g = zeros(max(max(elements)));
f_g = zeros(max(max(elements)),1);

%FEA code
tic
for el = 1:n_el %for number of elements
    % zero out K_el, f_el_omega, f_el_gamma
    K_el = zeros(n_node);
    f_el_omega = zeros(n_node,1);
    f_el_gamma = zeros(n_node,1);
    
    % construct local K and f
    for qp = 1:n_qp %for number of quadrature point
        xi = qp_coord(qp,1); %get current xi
        eta = qp_coord(qp,2); %get current eta
        %get N and delta N matrix
        N_e = [-0.25*(1-xi)*(1-eta)*(1+xi+eta),-0.25*(1+xi)*(1-eta)*(1-xi+eta),-0.25*(1+xi)*(1+eta)*(1-xi-eta),-0.25*(1-xi)*(1+eta)*(1+xi-eta),0.5*(1-xi)*(1+xi)*(1-eta),0.5*(1+xi)*(1-eta)*(1+eta),0.5*(1-xi)*(1+xi)*(1+eta),0.5*(1-xi)*(1-eta)*(1+eta)];
        delta_N_e = [-0.25*(1-eta)*(-2*xi-eta),-0.25*(1-eta)*(eta-2*xi),-0.25*(1+eta)*(-2*xi-eta),-0.25*(1+eta)*(eta-2*xi),-xi*(1-eta),0.5*(1-eta)*(1+eta),-xi*(eta+1),-0.5*(1-eta)*(1+eta);
            -0.25*(1-xi)*(-xi-2*eta),-0.25*(1+xi)*(xi-2*eta),-0.25*(1+xi)*(-xi-2*eta),-0.25*(1-xi)*(xi-2*eta),-0.5*(1-xi)*(1+xi),-eta*(xi+1),0.5*(1-xi)*(1+xi),-eta*(-xi+1)];
        %get J
        P = [nodes(elements(el,1),1),nodes(elements(el,1),2);
            nodes(elements(el,2),1),nodes(elements(el,2),2);
            nodes(elements(el,3),1),nodes(elements(el,3),2);
            nodes(elements(el,4),1),nodes(elements(el,4),2);
            nodes(elements(el,5),1),nodes(elements(el,5),2);
            nodes(elements(el,6),1),nodes(elements(el,6),2);
            nodes(elements(el,7),1),nodes(elements(el,7),2);
            nodes(elements(el,8),1),nodes(elements(el,8),2)];
        J = delta_N_e * P;
        J_inverse = inv(J);
        J_det = det(J);
        for i = 1:n_node 
            for j = 1:n_node
                K_el(i,j) = K_el(i,j)+k*transpose(J_inverse*[delta_N_e(1,i);delta_N_e(2,i)])*(J_inverse*[delta_N_e(1,j);delta_N_e(2,j)])*J_det;
            end
            f_el_omega(i) = f_el_omega(i)+s*transpose(N_e(i))*J_det;
        end
    end
    
    % Apply Neumann BC
    if NBC_flag == 1 % enable when NBC is needed
        if NBC_elem(el) == 1
            for qp = 1:n_qp_NBC  %for number of quadrature point
                xi = qp_coord_NBC(qp,1); %get current xi
                eta = qp_coord_NBC(qp,2); %get current eta
                N_e_NBC = [-0.25*(1-xi)*(1-eta)*(1+xi+eta),-0.25*(1+xi)*(1-eta)*(1-xi+eta),-0.25*(1+xi)*(1+eta)*(1-xi-eta),-0.25*(1-xi)*(1+eta)*(1+xi-eta),0.5*(1-xi)*(1+xi)*(1-eta),0.5*(1+xi)*(1-eta)*(1+eta),0.5*(1-xi)*(1+xi)*(1+eta),0.5*(1-xi)*(1-eta)*(1+eta)];
                for i = 1:n_node
                    f_el_gamma(i) = f_el_gamma(i) - q*N_e_NBC(i)*J_det_NBC;
                end
            end
        end
    end
    
    % Apply Dirichlet BC
    for i = gamma_T(el,:)
        if i ~= 0
            K_el(i,i) = C;
            f_el_omega(i) = C * T_bar(el);
        end
    end
    
    % Construct Global K and f
    K_g(elements(el,:),elements(el,:)) = K_g(elements(el,:),elements(el,:)) + K_el;
    f_g(elements(el,:)) = f_g(elements(el,:)) + f_el_omega + f_el_gamma;
    
end %end el = 1:n_el

% solve for T matrix
opts.SYM = true;
T = linsolve(K_g, f_g, opts);

% solve for heat flux q
q = zeros(2,n_el*n_qp);
x_qp = zeros(1,n_el*n_qp);
y_qp = zeros(1,n_el*n_qp);
index = 1;
for el = 1:n_el %for number of elements
    for qp = 1:n_qp %for number of quadrature point
        T_el = [T(elements(el,1));T(elements(el,2));T(elements(el,3));T(elements(el,4));
            T(elements(el,5));T(elements(el,6));T(elements(el,7));T(elements(el,8))];
        X_el = [nodes(elements(el,1),1);nodes(elements(el,2),1);nodes(elements(el,3),1);nodes(elements(el,4),1);
            nodes(elements(el,5),1);nodes(elements(el,6),1);nodes(elements(el,7),1);nodes(elements(el,8),1)];
        Y_el = [nodes(elements(el,1),2);nodes(elements(el,2),2);nodes(elements(el,3),2);nodes(elements(el,4),2);
            nodes(elements(el,5),2);nodes(elements(el,6),2);nodes(elements(el,7),2);nodes(elements(el,8),2)];
        xi = qp_coord(qp,1); %get current xi
        eta = qp_coord(qp,2); %get current eta
        delta_N_e = [-0.25*(1-eta)*(-2*xi-eta),-0.25*(1-eta)*(eta-2*xi),-0.25*(1+eta)*(-2*xi-eta),-0.25*(1+eta)*(eta-2*xi),-xi*(1-eta),0.5*(1-eta)*(1+eta),-xi*(eta+1),-0.5*(1-eta)*(1+eta);
            -0.25*(1-xi)*(-xi-2*eta),-0.25*(1+xi)*(xi-2*eta),-0.25*(1+xi)*(-xi-2*eta),-0.25*(1-xi)*(xi-2*eta),-0.5*(1-xi)*(1+xi),-eta*(xi+1),0.5*(1-xi)*(1+xi),-eta*(-xi+1)];
        %get J
        P = [nodes(elements(el,1),1),nodes(elements(el,1),2);
            nodes(elements(el,2),1),nodes(elements(el,2),2);
            nodes(elements(el,3),1),nodes(elements(el,3),2);
            nodes(elements(el,4),1),nodes(elements(el,4),2);
            nodes(elements(el,5),1),nodes(elements(el,5),2);
            nodes(elements(el,6),1),nodes(elements(el,6),2);
            nodes(elements(el,7),1),nodes(elements(el,7),2);
            nodes(elements(el,8),1),nodes(elements(el,8),2)];
        J = delta_N_e * P;
        J_inverse = inv(J);
        B = J_inverse*[-0.25*(1-eta)*(-2*xi-eta),-0.25*(1-eta)*(eta-2*xi),-0.25*(1+eta)*(-2*xi-eta),-0.25*(1+eta)*(eta-2*xi),-xi*(1-eta),0.5*(1-eta)*(1+eta),-xi*(eta+1),-0.5*(1-eta)*(1+eta);
                -0.25*(1-xi)*(-xi-2*eta),-0.25*(1+xi)*(xi-2*eta),-0.25*(1+xi)*(-xi-2*eta),-0.25*(1-xi)*(xi-2*eta),-0.5*(1-xi)*(1+xi),-eta*(xi+1),0.5*(1-xi)*(1+xi),-eta*(-xi+1)];
        N = [-0.25*(1-xi)*(1-eta)*(1+xi+eta),-0.25*(1+xi)*(1-eta)*(1-xi+eta),-0.25*(1+xi)*(1+eta)*(1-xi-eta),-0.25*(1-xi)*(1+eta)*(1+xi-eta),0.5*(1-xi)*(1+xi)*(1-eta),0.5*(1+xi)*(1-eta)*(1+eta),0.5*(1-xi)*(1+xi)*(1+eta),0.5*(1-xi)*(1-eta)*(1+eta)];
        q(:,index) = B * T_el*-k;
        x_qp(index) = N * X_el;
        y_qp(index) = N * Y_el;
        index = index+1;
    end
end
toc

%% Graphs and Plots 
[Xmg,Ymg] = meshgrid(linspace(-0.5,0.5,20),linspace(-0.5,0.5,20));
[Xq,Yq,Tq] = griddata(nodes(:,1),nodes(:,2),T,Xmg,Ymg);

figure('Name','Problem 2 Temperature Field','NumberTitle','off')
surf(Xq,Yq,zeros(size(Xq)),Tq)
view(2)
shading interp
a=colorbar;
title('Problem 2 Temperature Field')
xlabel('Position in meters')
ylabel('Position in meters')
a.Label.String = 'Temperature';

figure('Name','Problem 2 Heat Flux','NumberTitle','off')
quiver(x_qp,y_qp,q(1,:),q(2,:))
hold on
scatter(x_qp,y_qp)
title('Problem 2 Heat Flux')
xlabel('Position in meters')
ylabel('Position in meters')
hold off
    
