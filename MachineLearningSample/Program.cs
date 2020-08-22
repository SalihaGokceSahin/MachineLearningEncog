using Encog.MathUtil.Randomize;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.Engine.Network.Activation;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Structure;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.ML.Data;
using Encog.ML.Data.Basic;

namespace MachineLearningSample
{
    class Program
    {
        static void Main(string[] args)
        {
            #region dimensional arrays comment
            /*two dimensional array for summary,
            input array name is=x,
            output array name is=y*/
            #endregion
            #region dimensional arrays created for dataset
            double[][] x =
            {
               new double[]{0.1, 0.4},
               new double[]{0.3, 0.5},
               new double[]{0.5, 0.2},
               new double[]{0.7,0.3},
            };
            double[][] y =
            {
                new double[]{0.5},
                new double[]{0.8},
                new double[]{0.7},
                new double[]{1.0}
            };
            #endregion
            #region artificial neural network created
            //Layers:2 input, 5 neurons hidden layer, 1 output layer
            BasicNetwork network = new BasicNetwork();
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));//input neuron layer
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5));//hidden neuron layer
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 1));//output neuron layer
            network.Structure.FinalizeStructure();
            network.Reset();
            #endregion
            #region training
            IMLDataSet dataSet = new BasicMLDataSet(x,y);//filling data inside dataset
            ITrain learner = new Backpropagation(network, dataSet);
            for(int i = 0; i < 3000; i++)
            {
                learner.Iteration();
                //Console.WriteLine("error is:" + learner.Error);
            }
            #endregion
            #region testing
            foreach (BasicMLDataPair pair in dataSet)
            {
                IMLData result = network.Compute(pair.Input);
                Console.WriteLine(" {0} + {1} = {2} -> {3} ", pair.Input[0], pair.Input[1], pair.Ideal[0], result[0]);
            }
            Console.ReadKey();
            #endregion
        }
    }
}
