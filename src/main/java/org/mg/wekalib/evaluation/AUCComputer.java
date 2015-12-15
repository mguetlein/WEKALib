package org.mg.wekalib.evaluation;

import java.util.ArrayList;
import java.util.List;

import org.mg.wekalib.data.ArffWritable;
import org.mg.wekalib.data.ArffWriter;
import org.mg.wekalib.evaluation.alt.ExtendedEvaluation;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class AUCComputer
{
	public static void main(String[] args)
	{
		boolean actual[] = new boolean[] { true, true, true, false, false, false };
		boolean predicted[] = new boolean[] { true, true, true, false, false, true };
		double conf[] = new double[] { 0.6, 0.6, 0.8, 0.8, 0.8, 0.7 };
		System.out.println(compute(actual, predicted, conf));
	}

	public static double compute(boolean[] actual, boolean[] predicted, double conf[])
	{
		try
		{
			List<double[]> probs = new ArrayList<>();
			for (int i = 0; i < predicted.length; i++)
			{
				double p[] = new double[2];
				double prob = 0.5 + 0.5 * conf[i];
				if (predicted[i])
				{
					p[1] = prob;
					p[0] = 1 - prob;
				}
				else
				{
					p[0] = prob;
					p[1] = 1 - prob;
				}
				probs.add(p);
			}
			return compute(actual, probs);
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	/**
	 * probs: 2-dim array, [0] false, [1] true
	 */
	public static double compute(final boolean[] actual, final List<double[]> probs) throws Exception
	{
		Instances inst = ArffWriter.toInstances(new ArffWritable()
		{
			@Override
			public boolean isSparse()
			{
				return false;
			}

			@Override
			public boolean isInstanceWithoutAttributeValues(int instance)
			{
				return false;
			}

			@Override
			public String getRelationName()
			{
				return "bla";
			}

			@Override
			public int getNumInstances()
			{
				return actual.length;
			}

			@Override
			public int getNumAttributes()
			{
				return 1;
			}

			@Override
			public String getMissingValue(int attribute)
			{
				return null;
			}

			@Override
			public String getAttributeValueSpace(int attribute)
			{
				return "{0,1}";
			}

			@Override
			public String getAttributeValue(int instance, int attribute) throws Exception
			{
				return actual[instance] ? "1" : "0";
			}

			@Override
			public String getAttributeName(int attribute)
			{
				return "clazz";
			}

			@Override
			public List<String> getAdditionalInfo()
			{
				return null;
			}
		});
		inst.setClassIndex(0);
		Classifier cl = new Classifier()
		{
			int predCount = 0;

			@Override
			public Capabilities getCapabilities()
			{
				return null;
			}

			@Override
			public double[] distributionForInstance(Instance instance) throws Exception
			{
				return probs.get(predCount++);
			}

			@Override
			public double classifyInstance(Instance instance) throws Exception
			{
				throw new RuntimeException();
			}

			@Override
			public void buildClassifier(Instances data) throws Exception
			{
			}
		};
		Evaluation eval = new ExtendedEvaluation(inst);
		eval.evaluateModel(cl, inst);
		return eval.areaUnderROC(0);
	}
}
