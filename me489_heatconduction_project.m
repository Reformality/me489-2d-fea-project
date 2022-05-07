%-------------------------------------------------------------------------%
%                ME 489 Introduction to Finite Element Analysis           %
%                        Project function template                        %
%                                                                         %
% Note: 1) Use this function as a starting point for your code            %
%       2) Make sure all other functions that you need to run your code   %
%          are called from here. When the code is tested only this        %
%          function will be run.                                          %
%-------------------------------------------------------------------------%


clear;
clc;
disp('This project was done by')
student_name = 'Wu_Zekun';
disp(student_name)
fprintf('\n');
%%% Creates a directory with your name .
if (~exist(student_name,'dir'))
   % Command under Window
   system(['md', student_name]);
   % Command under MacOS
%    system(['mkdir ',student_name]);
end
    
% Problem 1:
problem1;


% Problem 2:
problem2;

% % Problem 3:
problem3_1;

problem3_2;

% % Problem 4:
% problem4;

