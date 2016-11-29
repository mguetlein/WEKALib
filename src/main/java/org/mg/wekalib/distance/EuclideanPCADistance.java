package org.mg.wekalib.distance;

import java.util.HashMap;
import java.util.Map;

import weka.attributeSelection.PrincipalComponents;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class EuclideanPCADistance implements Distance
{
	public static final long VERSION = 1;

	PrincipalComponents pca;
	EuclideanDistance dist;
	Instances transformedTrainingData;
	Map<Instances, Instances> transformedTestData = new HashMap<>();

	public String toString()
	{
		return this.getClass().toString() + "#" + VERSION;
	}

	public static EuclideanPCADistance fromString(String s)
	{
		if (s.startsWith(EuclideanPCADistance.class.toString()) && s.endsWith(VERSION + ""))
			return new EuclideanPCADistance();
		else
			return null;
	}

	@Override
	public void build(Instances train)
	{
		try
		{
			pca = new PrincipalComponents();
			pca.setCenterData(false);
			pca.setVarianceCovered(0.95);
			pca.buildEvaluator(train);
			transformedTrainingData = pca.transformedData(train);
			dist = new EuclideanDistance(transformedTrainingData);
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public double distance(int trainInstanceIdx1, int trainInstanceIdx2)
	{
		return distanceInternal(transformedTrainingData.get(trainInstanceIdx1),
				transformedTrainingData.get(trainInstanceIdx2));
	}

	@Override
	public double distance(Instance testInstance, int trainInstanceIdx)
	{
		return distanceInternal(transformTestInstance(testInstance),
				transformedTrainingData.get(trainInstanceIdx));
	}

	@Override
	public double distance(Instance testInstance1, Instance testInstance2)
	{
		return distanceInternal(transformTestInstance(testInstance1),
				transformTestInstance(testInstance2));
	}

	private Instance transformTestInstance(Instance testInstance)
	{
		try
		{
			Instances testData = testInstance.dataset();
			int testIdx = testData.indexOf(testInstance);
			if (testData.size() > 1 && testIdx != -1)
			{
				if (!transformedTestData.containsKey(testData))
					transformedTestData.put(testData, pca.transformedData(testData));
				return transformedTestData.get(testData).get(testIdx);
			}
			else
				return pca.convertInstance(testInstance);
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	private double distanceInternal(Instance transformed1, Instance transformed2)
	{
		try
		{
			return dist.distance(transformed1, transformed2);
		}
		catch (Exception e)
		{
			throw new RuntimeException(e);
		}
	}

	@Override
	public Double getMaxDistance()
	{
		return null;
	}

}
