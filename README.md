This small project was mostly done to better understand how Levenberg Marquardt and Simulated Annealing both work, and also become
more proficient in C#.NET

They are adaptations of MATLAB codes.

For Simulated annealing:  https://www.mathworks.com/matlabcentral/fileexchange/33109-simulated-annealing-optimization

For Levenberg-Marquardt: lm.m on page 7 of http://people.duke.edu/~hpgavin/ce281/lm.pdf

For LM Matrix operations, Math.Net.Numerics was used to make things faster. However, it would be interesting to write out Matrix inversion and Multiplications later on.

Program.cs does the following:

-implements a function with known parameters 

-samples it 

-adds random noise to these samples

-uses both schemes to try to find the parameters again.


Unfortunately results aren't satifactory possibly due to improvements in memory management being needed.

This code can still(and will) be improved. However, as of this first publication, I am happy to have been able to understand the steps of each algorithm.
