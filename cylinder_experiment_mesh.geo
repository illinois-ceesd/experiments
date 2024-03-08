//

lcar0 = 0.0005;
lcar1 = 0.0030;
lcar2 = 0.04;
lcar3 = 0.002;
lcar4 = 0.0040;

x=0.09;
y=0.09;
rad_out = 0.02;
r_glass = 125e-4;
t_glass = 15e-4;
l_glass = 0.085;
y_glass = 0.185;

Point(2)  = { -0.003,   0.0, 0.0, lcar0};
Point(3)  = { -0.0015,   0.0, 0.0, lcar0};
Point(4)  = {  0.0,   0.0, 0.0, lcar0};
Point(5)  = { 0.0015,   0.0, 0.0, lcar0};
Point(6)  = { 0.003,   0.0, 0.0, lcar0};
Point(7)  = {-x, -y, 0.0, lcar2};
Point(8)  = { x, -y, 0.0, lcar2};
Point(9)  = { x,  y, 0.0, lcar2};
Point(10) = {-x,  y, 0.0, lcar2};

Point(11) = {-0.01,  0.006, 0.0, lcar1};
Point(12) = { 0.01,  0.006, 0.0, lcar1};
Point(13) = { 0.01, -0.01, 0.0, lcar1};
Point(14) = {-0.01, -0.01, 0.0, lcar1};

Point(17)  = {-rad_out, -y, 0.0, lcar4};
Point(18)  = { rad_out, -y, 0.0, lcar4};

//Glass tube points
Point(30) = {r_glass,  y, 0.0, lcar3};
Point(31) = {r_glass,  y-l_glass, 0.0, lcar3};
Point(32) = {r_glass-t_glass,  y-l_glass, 0.0, lcar3};

Point(33) = {r_glass-t_glass,  y_glass, 0.0, lcar3};
Point(34) = {-r_glass+t_glass,  y_glass, 0.0, lcar3};

Point(35) = {-r_glass+t_glass,  y-l_glass, 0.0, lcar3};
Point(36) = {-r_glass,  y-l_glass, 0.0, lcar3};
Point(37) = {-r_glass,  y, 0.0, lcar3};

//Glass tube inside coarser mesh

Point(41) = {r_glass-t_glass - 5e-3,  y-0.01, 0.0, lcar4};
Point(42) = {-r_glass+t_glass + 5e-3,  y-0.01, 0.0, lcar4};
Point(43) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.02, 0.0, lcar4};
Point(44) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.02, 0.0, lcar4};

Point(45) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.035, 0.0, lcar4};
Point(46) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.035, 0.0, lcar4};
Point(47) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.05, 0.0, lcar4};
Point(48) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.05, 0.0, lcar4};

Point(49) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.065, 0.0, lcar4};
Point(50) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.065, 0.0, lcar4};

Point(51) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.08, 0.0, lcar4};
Point(52) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.08, 0.0, lcar4};

Point(53) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.095, 0.0, lcar4};
Point(54) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.095, 0.0, lcar4};

Point(55) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.11, 0.0, lcar4};
Point(56) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.11, 0.0, lcar4};

Point(57) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.125, 0.0, lcar4};
Point(58) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.125, 0.0, lcar4};

Point(59) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.140, 0.0, lcar4};
Point(60) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.140, 0.0, lcar4};

Point(61) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.155, 0.0, lcar4};
Point(62) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.155, 0.0, lcar4};
Point(63) = {-r_glass+t_glass + 5e-3,  y-l_glass + 0.170, 0.0, lcar4};
Point(64) = {r_glass-t_glass - 5e-3,  y-l_glass + 0.170, 0.0, lcar4};
//Point(48) = {r_glass-t_glass - 3e-3,  y-l_glass + 0.065, 0.0, lcar4};
//Define bounding box edges
Line(1) = {7, 17};
Line(31) = {17, 18};
Line(2) = {18,8};
Line(3) = {8,9};
Line(4) = {10,7};

Line(5) = {2, 3};
Circle(6) = {3, 4, 5};
Line(7) = {5, 6};
Circle(8) = {5, 4, 3};
Circle(9) = {2, 4, 6};
Circle(10) = {6, 4, 2};

//Line(11) = {11, 12};
//Line(12) = {12, 13};
//Line(13) = {13, 14};
//Line(14) = {14, 11};

//glass tube lines

Line(20) = {9, 30};
Line(21) = {30, 31};
Line(22) = {31, 32};
Line(23) = {32, 33};
Line(24) = {33, 34};
Line(25) = {34, 35};
Line(26) = {35, 36};
Line(27) = {36, 37};
Line(28) = {37, 10};

Transfinite Line { 6} = 16 Using Progression 1.0;
Transfinite Line { 8} = 16 Using Progression 1.0;
Transfinite Line { 9} = 16 Using Progression 1.0;
Transfinite Line {10} = 16 Using Progression 1.0;
Transfinite Line { 5} = 4 Using Progression 1.0;
Transfinite Line { 7} = 4 Using Progression 1.0;


Line Loop(101) = {1,31,2,3,20:28,4,9,10};
Line Loop(103) = {5,6,7,-9};
Line Loop(104) = {-8,5,10,7};


//Define unstructured far field mesh zone
Plane Surface(201) = {-101};
Point{11:14} In Surface{201};
Point{41:64} In Surface{201};

Plane Surface(203) = {103};
Transfinite Surface{203} Alternate;

Plane Surface(204) = {-104};
Transfinite Surface{204} Alternate;


Physical Line("inflow") = {24};
Physical Line("glasstube") = {21,22,23,25,26,27};
Physical Line("surface") = {6,8};
Physical Line("outflow") = {31};
Physical Line("sidewall")={1,2,3,4,28,20};
Physical Surface("fluid") = {201,203,204};



