clc;

L = [1 1 1 0 0 0];

data = diag(L)

data( L == 0, : ) = [];  %rows
data( :, L == 0 ) = [];  %columns

data

v = [1 2 3 4 5 6]

v(:, L==0) = []

v'


x = [1 2; 3 4]

x^-0.5 * x^-0.5

x^-1