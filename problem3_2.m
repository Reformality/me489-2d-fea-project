clear;
clc;

nodes = [  0.0000,    0.0000;
           0.0000,   -0.1000;
           0.4155,   -0.1000;
           0.3000,    0.0000;
           0.4732,    0.0000;
           0.0000,   -0.3000;
          -0.4155,   -0.1000;
          -0.3000,   -0.3000;
           0.3000,   -0.3000;
          -0.3000,    0.0000;
          -0.4732,    0.0000;
           0.0000,   -0.0500;
           0.1039,   -0.1000;
           0.2077,   -0.1000;
           0.3116,   -0.1000;
           0.4443,   -0.0500;
           0.1500,    0.0000;
           0.3866,    0.0000;
           0.0000,   -0.2000;
          -0.1039,   -0.1000;
          -0.2077,   -0.1000;
          -0.3116,   -0.1000;
          -0.3577,   -0.2000;
          -0.2250,   -0.3000;
          -0.1500,   -0.3000;
          -0.0750,   -0.3000;
           0.0750,   -0.3000;
           0.1500,   -0.3000;
           0.2250,   -0.3000;
           0.3577,   -0.2000;
          -0.1500,    0.0000;
          -0.3866,	  0.0000;
          -0.4443,	 -0.0500;
           0.1269,	 -0.0500;
           0.2539,	 -0.0500;
           0.3491,	 -0.0500;
          -0.0894,	 -0.2000;
          -0.1789,	 -0.2000;
          -0.2683,	 -0.2000;
           0.0894,	 -0.2000;
           0.1789,	 -0.2000;
           0.2683,	 -0.2000;
          -0.1269,	 -0.0500;
          -0.2539,	 -0.0500;
          -0.3491,	 -0.0500;
           0.0000,	 -0.0250;
           0.0635,	 -0.0500;
           0.1385,	 -0.0250;
           0.0750,	  0.0000;
           0.0000,	 -0.0750;
           0.0519,	 -0.1000;
           0.1154,	 -0.0750;
           0.1904,	 -0.0500;
           0.2769,	 -0.0250;
           0.2250,	  0.0000;
           0.1558,	 -0.1000;
           0.2308,	 -0.0750;
           0.3015,	 -0.0500;
           0.3679,	 -0.0250;
           0.3433,	  0.0000;
           0.2597,	 -0.1000;
           0.3303,	 -0.0750;
           0.3967,	 -0.0500;
           0.4588,	 -0.0250;
           0.4299,	  0.0000;
           0.3635,	 -0.1000;
           0.4299,	 -0.0750;
           0.0000,	 -0.2500;
          -0.0447,	 -0.2000;
          -0.0822,	 -0.2500;
          -0.0375,	 -0.3000;
           0.0000,	 -0.1500;
          -0.0519,	 -0.1000;
          -0.0967,	 -0.1500;
          -0.1341,	 -0.2000;
          -0.1644,	 -0.2500;
          -0.1125,	 -0.3000;
          -0.1558,	 -0.1000;
          -0.1933,	 -0.1500;
          -0.2236,	 -0.2000;
          -0.2466,	 -0.2500;
          -0.1875,	 -0.3000;
          -0.2597,	 -0.1000;
          -0.2900,	 -0.1500;
          -0.3130,	 -0.2000;
          -0.3289,	 -0.2500;
          -0.2625,	 -0.3000;
          -0.3635,	 -0.1000;
          -0.3866,	 -0.1500;
           0.0447,	 -0.2000;
           0.0967,	 -0.1500;
           0.0375,	 -0.3000;
           0.0822,	 -0.2500;
           0.1341,	 -0.2000;
           0.1933,	 -0.1500;
           0.1125,	 -0.3000;
           0.1644,	 -0.2500;
           0.2236,	 -0.2000;
           0.2900,	 -0.1500;
           0.1875,	 -0.3000;
           0.2466,	 -0.2500;
           0.3130,	 -0.2000;
           0.3866,	 -0.1500;
           0.2625,	 -0.3000;
           0.3289,	 -0.2500;
          -0.0635,	 -0.0500;
          -0.1154,	 -0.0750;
          -0.0750,	  0.0000;
          -0.1385,	 -0.0250;
          -0.1904,	 -0.0500;
          -0.2308,	 -0.0750;
          -0.2250,	  0.0000;
          -0.2769,	 -0.0250;
          -0.3015,	 -0.0500;
          -0.3303,	 -0.0750;
          -0.3433,	  0.0000;
          -0.3679,	 -0.0250;
          -0.3967,	 -0.0500;
          -0.4299,	 -0.0750;
          -0.4299,	  0.0000;
          -0.4588,	 -0.0250 ];

      
      
elements = [  1,    12,    34,    17,    46,    47,    48,    49;
             12,     2,    13,    34,    50,    51,    52,    47;
             17,    34,    35,     4,    48,    53,    54,    55;
             34,    13,    14,    35,    52,    56,    57,    53;
              4,    35,    36,    18,    54,    58,    59,    60;
             35,    14,    15,    36,    57,    61,    62,    58;
             18,    36,    16,     5,    59,    63,    64,    65;
             36,    15,     3,    16,    62,    66,    67,    63;
              6,    19,    37,    26,    68,    69,    70,    71;
             19,     2,    20,    37,    72,    73,    74,    69;
             26,    37,    38,    25,    70,    75,    76,    77;
             37,    20,    21,    38,    74,    78,    79,    75;
             25,    38,    39,    24,    76,    80,    81,    82;
             38,    21,    22,    39,    79,    83,    84,    80;
             24,    39,    23,     8,    81,    85,    86,    87;
             39,    22,     7,    23,    84,    88,    89,    85;
              2,    19,    40,    13,    72,    90,    91,    51;
             19,     6,    27,    40,    68,    92,    93,    90;
             13,    40,    41,    14,    91,    94,    95,    56;
             40,    27,    28,    41,    93,    96,    97,    94;
             14,    41,    42,    15,    95,    98,    99,    61;
             41,    28,    29,    42,    97,   100,   101,    98;
             15,    42,    30,     3,    99,   102,   103,    66;
             42,    29,     9,    30,   101,   104,   105,   102;
              2,    12,    43,    20,    50,   106,   107,    73;
             12,     1,    31,    43,    46,   108,   109,   106;
             20,    43,    44,    21,   107,   110,   111,    78;
             43,    31,    10,    44,   109,   112,   113,   110;
             21,    44,    45,    22,   111,   114,   115,    83;
             44,    10,    32,    45,   113,   116,   117,   114;
             22,    45,    33,     7,   115,   118,   119,    88;
             45,    32,    11,    33,   117,   120,   121,   118 ];
%% 
%-------------------------------------------------------------------------%
%                ME 489 Introduction to Finite Element Analysis           %
%                        Project function template                        %
%                                                                         %
% Note: 1) Use this function as a starting point for your code            %
%       2) Make sure all other functions that you need to run your code   %
%          are called from here. When the code is tested only this        %
%          function will be run.                                          %
%-------------------------------------------------------------------------%

% % Problem 3:
NBC_flag = 0;
s = 0;

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
    % get k value
    if max(nodes(elements(el,:),2)) == 0
        k = 1;
    else
        k = 2;
    end
    
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
    for i = 1:n_node
        if nodes(elements(el,i),2) == -0.3
            T_bar = 100;
            K_el(i,i) = C;
            f_el_omega(i) = C * T_bar;
        elseif nodes(elements(el,i),2) == 0
            T_bar = 25;
            K_el(i,i) = C;
            f_el_omega(i) = C * T_bar; 
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
    % get k value
    if max(nodes(elements(el,:),2)) == 0
        k = 1;
    else
        k = 2;
    end
    
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
%meshgrid, griddata, surf
[Xmg,Ymg] = meshgrid(linspace(-0.5,0.5,1000),linspace(-0.3,0,1000));
[Xq,Yq,Tq] = griddata(nodes(:,1),nodes(:,2),T,Xmg,Ymg);

figure('Name','Problem 3.2 Temperature Field','NumberTitle','off')
surf(Xq,Yq,zeros(size(Xq)),Tq)
view(2)
shading interp
a=colorbar;
title('Problem 3.2 Temperature Field')
xlabel('Position in meters')
ylabel('Position in meters')
a.Label.String = 'Temperature';

figure('Name','Problem 3.2 Heat Flux','NumberTitle','off')
quiver(x_qp,y_qp,q(1,:),q(2,:))
hold on
scatter(x_qp,y_qp)
title('Problem 3.2 Heat Flux')
xlabel('Position in meters')
ylabel('Position in meters')
hold off