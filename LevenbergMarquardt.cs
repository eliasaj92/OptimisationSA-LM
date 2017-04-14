using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace LMA
{
    class LevenbergMarquardt
    {
        static int iteration;

        public static void lm(Func<List<double>, double[], List<double>> f, ref double[] p, List<double> t, List<double> y_dat,
            double[] lower, double[] upper, int MaxIter = 100, double tolG = 0, double tolP = 0,
            double tolChi = 0, double tolA = 0, double lambda_0 = 5, double lambdaUp = 11, double lambdaDown = 9)
        {
            iteration = 0;
            int nPar = p.Length;
            int nPnt = y_dat.Count;
            double[] p_try = new double[p.Length];
            double[] p_old = new double[p.Length];
            Matrix<double> y_datm = Vector<double>.Build.DenseOfEnumerable(y_dat).ToColumnMatrix(); 
            Matrix<double> y_old = Matrix<double>.Build.DenseOfMatrix(y_datm);
            Matrix<double> y_hat = Matrix<double>.Build.DenseOfMatrix(y_datm);
            Matrix<double> delta_y = Matrix<double>.Build.Dense(nPnt, 1);
            double X2 = double.PositiveInfinity;
            double X2_old = double.PositiveInfinity;
            double X2_try = double.PositiveInfinity;
            double X21 = -1;
            double dX2 = 0;
            Matrix<double> J = Matrix<double>.Build.Dense(nPar, nPnt);
            int DoF = nPnt - nPar + 1;
            Matrix<double> JtJ = Matrix<double>.Build.Dense(nPnt, nPnt);
            Matrix<double> Jtdy = Matrix<double>.Build.Dense(nPnt, 1);
            bool stop = false;
            double denumRho = 1;
            double rho = 0;

            lm_matx(out JtJ, out Jtdy, out X2, out y_hat, ref J, f, t, p_old, y_old, 1, p, y_datm);

            if (Jtdy.ReduceRows((v1, v2) => v1.AbsoluteMaximum() > v2.AbsoluteMaximum() ? v1 : v2).AbsoluteMaximum() < tolG)
            {
                stop = true;
            }

            double lambda = lambda_0;
            X2_old = X2;

            while (!stop && iteration < MaxIter)
            {
                //get parameters update
                Matrix<double> dJtJ = Matrix<double>.Build.DenseOfDiagonalVector(JtJ.Diagonal().Multiply(lambda));
                Matrix<double> h =  (JtJ + dJtJ).Inverse() * Jtdy;

                //update paramters and check they are in bounds
                for (int i = 0; i < p.Length; i++)
                {
                    p_try[i] = p[i] + h.ToArray()[i, 0];
                    p_try[i] = p_try[i] > lower[i] ? p_try[i] : lower[i];
                    p_try[i] = p_try[i] < upper[i] ? p_try[i] : upper[i];
                }

                y_hat = Vector<double>.Build.DenseOfEnumerable(f(t, p_try)).ToColumnMatrix();
                delta_y = y_datm - y_hat;
                X2_try = (delta_y.Transpose() * delta_y)[0,0];

                denumRho = (h.Transpose() * (h.Multiply(lambda) + Jtdy))[0,0];
                rho = (X2 - X2_try) / denumRho;

                if(rho> tolA)
                {
                    dX2 = X2 - X2_old;
                    X2_old = X2;
                    y_old = Matrix<double>.Build.DenseOfMatrix(y_hat);
                    for (int i = 0; i < p.Length; i++)
                    {
                        p_old[i] = p[i];
                        p[i] = p_try[i];
                    }

                    lm_matx(out JtJ, out Jtdy, out X2, out y_hat, ref J, f, t, p_old, y_old, 1, p, y_datm);

                    lambda = Math.Max(lambda / lambdaDown, 1E-7);
                }
                else
                {
                    X2 = X2_old;

                    if(iteration % 2*nPar !=0)
                    {
                        X21 = -1;
                        lm_matx(out JtJ, out Jtdy, out X21, out y_hat, ref J, f, t, p_old, y_old, 1, p, y_datm);    
                    }

                    lambda = Math.Min(lambda * lambdaUp, 1E7);
                }

                iteration++;

                if (Jtdy.ReduceRows((v1, v2) => v1.AbsoluteMaximum() > v2.AbsoluteMaximum() ? v1 : v2).AbsoluteMaximum() < tolG && iteration>2)
                {
                    stop = true;
                }
                for(int i = 0; i<nPar; i++)
                {
                    if(Math.Abs(h[i,0]/p[i])< tolP && iteration >2)
                    {
                        stop = true;
                    }
                }

                if(X2/DoF < tolChi && iteration >3)
                {
                    stop = true;
                }

            }
        }

        public static void lm_matx(out Matrix<double> JtJ, out Matrix<double> Jtdy, out double X2, out Matrix<double> y_hat,
            ref Matrix<double> J, Func<List<double>, double[], List<double>> f, List<double> t, double[] p_old, Matrix<double> y_old,
            double dX2, double[] p, Matrix<double> y_dat)
        {
            int nPar = p.Length;
            int nPnt = y_dat.RowCount;
            JtJ = Matrix<double>.Build.Dense(nPnt, nPnt);
            Jtdy = Vector<double>.Build.Dense(nPnt).ToColumnMatrix();
            y_hat = Vector<double>.Build.DenseOfEnumerable(f(t, p)).ToColumnMatrix();
            Matrix<double> yhv = y_hat;
            Matrix<double> ydv = y_dat;
            Matrix<double> delta_y = ydv - yhv;

            if (iteration % (2 * nPar) != 0 || dX2 > 0)
            {
                FiniteDifference(out J, f, t, ref p, yhv);
            }
            else
            {
                Broyden(p_old, p, ref J, y_old, y_hat);
            }
            Matrix<double> X2m = (delta_y.Transpose() * delta_y);
            X2 = (delta_y.Transpose() * delta_y).ToArray()[0, 0];
            JtJ = J.Transpose() * (J);
            Jtdy = J.Transpose() * delta_y;
        }

        public static void FiniteDifference(out Matrix<double> J, Func<List<double>, double[], List<double>> f, List<double> t,
           ref double[] p, Matrix<double> y)
        {
            //only central difference implemented (dp>0)
            int nPnt = y.RowCount;
            int nPar = p.Length;
            double dp = 10;
            Matrix<double> y1 = Matrix<double>.Build.DenseOfMatrix(y);
            Matrix<double> y2 = Matrix<double>.Build.DenseOfMatrix(y);

            J = Matrix<double>.Build.Dense(nPnt, nPar);
            Matrix<double> pm = Vector<double>.Build.DenseOfArray(p).ToColumnMatrix();
            Matrix<double> ps = pm;
            double[] psa = p;

            double[] del = new double[nPar];

            for (int j = 0; j < nPar; j++)
            {
                del[j] = dp * (Math.Abs(p[j]) + 1);
                p[j] = psa[j] + del[j];

                if (del[j] != 0)
                {
                    y1 = Vector<double>.Build.DenseOfEnumerable(f(t, p)).ToColumnMatrix();


                    //central difference as dp>0
                    p[j] = psa[j] - del[j];
                   
                    y2 = Vector<double>.Build.DenseOfEnumerable(f(t, p)).ToColumnMatrix();
                    for (int i = 0; i < nPnt; i++)
                    {
                        J[i, j] = (y1[i,0] - y2[i,0]) / (2 * del[j]);
                    }
                }
                p[j] = psa[j];
            }

        }

        public static void Broyden(double[] p_old, double[] p, ref Matrix<double> J, Matrix<double> y_old, Matrix<double> y)
        {
            Matrix<double> pm = Vector<double>.Build.DenseOfArray(p).ToColumnMatrix();
            Matrix<double> pmo = Vector<double>.Build.DenseOfArray(p_old).ToColumnMatrix();
            Matrix<double> ym = y;
            Matrix<double> ymo = y_old;

            Matrix<double> h = pm - pmo;

            J = J + (ym - ymo - J * h) * h.Transpose() * (h.Transpose() * h).Inverse();
        }

    }
}
