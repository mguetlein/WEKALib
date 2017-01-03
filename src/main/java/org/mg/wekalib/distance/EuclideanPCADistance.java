package org.mg.wekalib.distance;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import weka.attributeSelection.PrincipalComponents;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.neighboursearch.PerformanceStats;

public class EuclideanPCADistance implements Distance, DistanceFunction, Serializable
{
	public static final long VERSION = 1;

	/**
	 * 
	 */
	private static final long serialVersionUID = VERSION;

	PrincipalComponents pca;
	EuclideanDistance dist;
	Instances trainingData;
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
			trainingData = train;
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

	// weka methods implementing Distance Function

	@Override
	public String getAttributeIndices()
	{
		return "all";
	}

	@Override
	public Instances getInstances()
	{
		return trainingData;
	}

	@Override
	public boolean getInvertSelection()
	{
		return false;
	}

	@Override
	public String[] getOptions()
	{
		return new String[0];
	}

	@Override
	public Enumeration<Option> listOptions()
	{
		return null;
	}

	@Override
	public void postProcessDistances(double[] distances)
	{
	}

	@Override
	public void setAttributeIndices(String value)
	{
		throw new IllegalStateException("not implemented");
	}

	@Override
	public void setInstances(Instances insts)
	{
		build(insts);
	}

	@Override
	public void setInvertSelection(boolean value)
	{
		throw new IllegalStateException("not implemented");
	}

	@Override
	public void setOptions(String[] options) throws Exception
	{
		throw new IllegalStateException("not implemented");
	}

	@Override
	public void update(Instance ins)
	{
		throw new IllegalStateException("not implemented");
	}

	@Override
	public double distance(Instance first, Instance second, PerformanceStats stats) throws Exception
	{
		return distance(first, second, Double.POSITIVE_INFINITY, stats);
	}

	@Override
	public double distance(Instance first, Instance second, double cutOffValue)
	{
		return distance(first, second, cutOffValue, null);
	}

	@Override
	public double distance(Instance first, Instance second, double cutOffValue,
			PerformanceStats stats)
	{
		double distance = distance(first, second);
		if (stats != null)
			stats.incrCoordCount();
		if (distance > cutOffValue)
			return Double.POSITIVE_INFINITY;
		return distance;
	}

	@Override
	public void clean()
	{
		trainingData = null;
		pca = null;
		transformedTrainingData = null;
		dist = null;
	}

}
