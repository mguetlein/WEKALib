package org.mg.wekalib.classifier;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.classifiers.ConditionalDensityEstimator;
import weka.classifiers.meta.Bagging;
import weka.core.Instance;
import weka.core.Utils;

public class MyBagging extends Bagging implements ConditionalDensityEstimator
{
	Instance lastClassified;
	double variance;

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		DescriptiveStatistics stats = new DescriptiveStatistics();

		double[] sums = new double[instance.numClasses()], newProbs;

		double numPreds = 0;
		for (int i = 0; i < m_NumIterations; i++)
		{
			if (instance.classAttribute().isNumeric() == true)
			{
				double pred = m_Classifiers[i].classifyInstance(instance);
				if (!Utils.isMissingValue(pred))
				{
					stats.addValue(pred);
					sums[0] += pred;
					numPreds++;
				}
			}
			else
			{
				newProbs = m_Classifiers[i].distributionForInstance(instance);
				for (int j = 0; j < newProbs.length; j++)
					sums[j] += newProbs[j];
			}
		}

		lastClassified = instance;
		variance = stats.getVariance();

		if (instance.classAttribute().isNumeric() == true)
		{
			if (numPreds == 0)
			{
				sums[0] = Utils.missingValue();
			}
			else
			{
				sums[0] /= numPreds;
			}
			return sums;
		}
		else if (Utils.eq(Utils.sum(sums), 0))
		{
			return sums;
		}
		else
		{
			Utils.normalize(sums);
			return sums;
		}
	}

	@Override
	public double logDensity(Instance instance, double value) throws Exception
	{
		if (lastClassified != instance)
			distributionForInstance(instance);
		return -1 * variance;
	}
}
