package org.mg.wekalib.appdomain;

import weka.core.Instance;
import weka.core.Instances;

public class TanimotoAppDomainModel extends NNDistanceBasedAppDomainModel
{
	@Override
	protected void buildInternal(Instances trainingData)
	{
		for (int i = 0; i < trainingData.numAttributes(); i++)
		{
			if (trainingData.classIndex() == i)
				continue;
			if (!trainingData.attribute(i).isNominal())
				throw new IllegalArgumentException("not nominal : " + trainingData.attribute(i));
			if (trainingData.attribute(i).numValues() != 2)
				throw new IllegalArgumentException("not binary : " + trainingData.attribute(i));
			//			for (int j = 0; j < trainingData.numInstances(); j++)
			//			{
			//				double val = trainingData.get(j).value(i);
			//				if (!Double.isNaN(val) && val != 0.0 && val != 1.0)
			//					throw new IllegalArgumentException(val + "");
			//			}
		}
	}

	@Override
	protected double computeDistance(Instance i1, Instance i2)
	{
		double and = 0;
		double or = 0;
		for (int i = 0; i < i1.numAttributes(); i++)
		{
			if (i1.classIndex() == i)
				continue;
			if (i1.value(i) == 1.0)
			{
				if (i2.value(i) == 1.0)
					and++;
				or++;
			}
			else if (i2.value(i) == 1.0)
			{
				or++;
			}
		}
		if (or == 0)
			return 1.0;
		else
			return 1.0 - (and / or);
	}

}
