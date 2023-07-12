//

lcar0 = 0.00015;
lcar1 = 0.0020;
lcar2 = 0.0003;


Point(7)  = {-0.00015, -0.00300, 0.0, lcar2};
Point(8)  = { 0.00015, -0.00300, 0.0, lcar2};
Point(9)  = { 0.00015,  0.00300, 0.0, lcar2};
Point(10) = {-0.00015,  0.00300, 0.0, lcar2};


//Define bounding box edges
Line(1) = {7, 8};
Line(2) = {8, 9};
Line(3) = {9,10};
Line(4) = {10,7};

Transfinite Line { 6} = 41 Using Progression 1.0;
Transfinite Line { 8} = 41 Using Progression 1.0;
Transfinite Line { 9} = 41 Using Progression 1.0;
Transfinite Line {10} = 41 Using Progression 1.0;

Line Loop(101) = {1,2,3,4};
//Line Loop(102) = {9,10,11,12,13,14};
//Line Loop(103) = {5,6,7,-9};
//Line Loop(104) = {-8,5,10,7};


//Define unstructured far field mesh zone
Plane Surface(201) = {-101};
Transfinite Surface{201} Alternate;

Physical Line("inflow") = {4};

Physical Line("surface") = {2};
Physical Line("sidewall")={1,3};
Physical Surface("fluid") = {201};

