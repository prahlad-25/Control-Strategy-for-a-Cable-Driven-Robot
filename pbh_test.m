close all;
clear all; 
clc;

rank_C = rank([-A B*W]);

disp(['Controllability matrix rank: ', num2str(rank_C)]);