package org.mg.wekalib.classifier;

import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;

import weka.classifiers.RandomizableClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class RandomClassifier extends RandomizableClassifier
{
	private static final long serialVersionUID = 1L;

	private Random random;

	public RandomClassifier()
	{
	}

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		random = new Random(m_Seed);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception
	{
		double[] dist = new double[instance.numClasses()];
		if (dist.length != 2)
			throw new NotImplementedException("to be added");
		dist[0] = random.nextDouble();
		dist[1] = 1.0 - dist[0];
		return dist;
	}

}
