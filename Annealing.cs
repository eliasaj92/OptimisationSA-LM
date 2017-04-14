using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace simulated_annealing
{
    class Annealing
    {
        private static readonly Random random = new Random();

        /// <summary>
        /// Implements the Simulated Annealing based on the Metropolis algorithm
        /// This code is a C#.NET implementation of the Matlab code obtained from: https://www.mathworks.com/matlabcentral/fileexchange/33109-simulated-annealing-optimization
        /// </summary>
        /// <param name="f"> cost function in the optimization </param>
        /// <param name="x0"> the parameters to be optimized </param>
        /// <param name="lower"> lower bounds for the parameter values. Array should be of same size</param>
        /// <param name="upper"></param>
        /// <param name="maxT">Maximum temperature the algoirthm reaches</param>
        /// <param name="tolerance"> tolerance value for which the algorithm stops</param>
        public static void SimulatedAnnealing(Func<double[], double> f, ref double[] x0, double[] lower, double[] upper, int maxT = 100, double tolerance = 0.00000000000000000001)
        {

            
            double[] x = x0;
            double[] dx = new double[x0.Length];
            double[] x1 = new double[x0.Length];
            double[] temp = new double[x0.Length];

            double fx = f(x);
            double f0 = fx;
            double fx1;
            double df;

            double inverseT;
            double mu;

            double e;

            for ( int m = 0; m<=maxT; m++)
            {
                inverseT = (m+0.0) / maxT;
                mu = Math.Pow(10.0, inverseT*100.0);


                //for every temperature try to find the best solution 500 times
                for (int k = 0; k <= 500; k++)
                {
                    //used parallelization for increased performance as each parameter is treated independently of the other
                    Parallel.For(0, dx.Length, i =>
                    {
                        temp[i] = 2 * random.NextDouble();
                        double r = muInv(temp[i] - 1, mu);
                        dx[i] =  r * (upper[i] - lower[i]);
                        x1[i] = x[i] + dx[i];

                        //following is to keep the values of x1[i] in between the bounds lower[] and upper[]
                        x1[i] = Convert.ToDouble(x1[i] < lower[i]) * lower[i]  //if x1[i] <lower[i], make it equal to lower[i]
                          + Convert.ToDouble(lower[i] <= x1[i]) * Convert.ToDouble(upper[i] >= x1[i]) * x1[i] //if x1[i] in between bounds, then keep its value
                          + Convert.ToDouble(upper[i] < x1[i]) * upper[i]; //if x1[i]>upper[i] then set it to upper[i]
                                                                           });
                    fx1 = f(x1);
                    df = fx - fx1;

                    //now to use the Metropolis criteria
                    e = Math.Exp(inverseT * df / (Math.Abs(fx) + Double.Epsilon) / tolerance);
                    if((df>0) || (e> random.NextDouble()) )
                    {
                        x = x1;
                        fx = fx1;
                    }

                    if (fx< f0)
                    {
                        x0 = x1;
                        f0 = fx1;
                    }
                }
            }
        }

        /// <summary>
        /// Gets step values for the parameters based on mu, which itself is based on temperature
        /// </summary>
        /// <param name="y">the parameter</param>
        /// <param name="mu">the mu value</param>
        /// <returns></returns>
        static double muInv(double y, double mu)
        {
            double temp = Math.Sign(y) * (Math.Pow(1 + mu, Math.Abs(y))-1 ) / mu;
            return temp;
            #region parallel for loop
            //Parallel.For(0, y.Length, i => 
            //{ x[i]  = Math.Sign(y[i]) * (Math.Pow(1 + mu, Math.Abs(y[i])) - 1) / mu; });
            #endregion
            #region sequential for loop
            //for(int i = 0; i<y.Length; i++)
            //{
            //    x[i] = Math.Sign(y[i]) * (Math.Pow(1 + mu, Math.Abs(y[i])) - 1) / mu;
            //}
            #endregion
        }

        
    }
}
