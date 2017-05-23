using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using simulated_annealing;

namespace LMA
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rand = new Random();
            int nPnt = 100;
            List<double> t = new List<double>(nPnt);
            for (int i = 1; i <= nPnt; i++)
            {
                t.Add(i);
            }
            double[] p_true = new double[4]
            {
                20,
                10,
                1,
                50,
            };
            List<double> ty_dat = lm_f(t, p_true);
            List<double> y_dat = new List<double>(nPnt);
            for (int i = 0; i < nPnt; i++)
            {
                y_dat.Add(GaussianNoise(rand, ty_dat.ElementAt(i), 5));
            }

            double[] p_init = new double[4]
            {
                20,
                2,
                0.2,
                10
            };


            double[] p_min = new double[4];
            double[] p_max = new double[4];

            for (int j = 0; j < 4; j++)
            {
                p_min[j] = -10 * Math.Abs(p_init[j]);
                p_max[j] = 10 * Math.Abs(p_init[j]);
            }

            //Annealing.SimulatedAnnealing(lm_MSE(lm_f, t, y_dat), ref p_init, p_min, p_max, 8, 0);

            //for (int j = 0; j < 4; j++)
            //{
            //    Console.WriteLine(p_init[j]);
            //}
            ////Console.Read();
            LevenbergMarquardt.lm(lm_f, ref p_init, t, y_dat, p_min, p_max, 2000, 0, 0, 0, 0, 5, 11, 9);

            for (int j = 0; j < 4; j++)
            {
                Console.WriteLine(p_init[j]);
            }
            Console.Read();
        }

        public static double GaussianNoise(Random rand, double mean, double stdDev)
        {
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal =
                         mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)
            return randNormal;
        }

        static List<double> lm_f(List<double> x, double[] p)
        {
            List<double> y = new List<double>(x.Count);
            double temp = 0;
            for (int i = 0; i < x.Count; i++)
            {
                temp = p[0] * Math.Exp(-x.ElementAt(i) / p[1]) + p[2] * Math.Exp(-x.ElementAt(i) / p[3]);
                y.Add(temp);
            }
            return y;
        }

        static Func<double[], double> lm_MSE(Func<List<double>, double[], List<double>> lm_f, List<double> t, List<double> y_dat)
        {
            Func<double[], double> temp = delegate (double[] x)
            {
                List<double> y = lm_f(t, x);
                double mse = 0;
                int n = y.Count;
                for (int i = 0; i < n; i++)
                {
                    mse += (y.ElementAt(i) - y_dat.ElementAt(i)) * (y.ElementAt(i) - y_dat.ElementAt(i)) / n;
                }
                return mse;
            };

            return temp;
        }
    }
}
