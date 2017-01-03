package org.mg.wekalib.distance;

import java.io.Serializable;
import java.util.Enumeration;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.neighboursearch.PerformanceStats;

public class EuclideanDistance implements Distance, DistanceFunction, Serializable
{
	public static final long VERSION = 1;

	/**
	 * 
	 */
	private static final long serialVersionUID = VERSION;

	weka.core.EuclideanDistance dist = new weka.core.EuclideanDistance();

	public String toString()
	{
		return this.getClass().toString() + "#" + VERSION;
	}

	public static EuclideanDistance fromString(String s)
	{
		if (s.startsWith(EuclideanDistance.class.toString()) && s.endsWith(VERSION + ""))
			return new EuclideanDistance();
		else
			return null;
	}

	@Override
	public void build(Instances train)
	{
		dist.setInstances(train);
	}

	@Override
	public double distance(int trainInstanceIdx1, int trainInstanceIdx2)
	{
		return distance(dist.getInstances().get(trainInstanceIdx1),
				dist.getInstances().get(trainInstanceIdx2));
	}

	@Override
	public double distance(Instance testInstance, int trainInstanceIdx)
	{
		return distance(testInstance, dist.getInstances().get(trainInstanceIdx));
	}

	@Override
	public double distance(Instance testInstance1, Instance testInstance2)
	{
		return dist.distance(testInstance1, testInstance2);
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
		return dist.getAttributeIndices();
	}

	@Override
	public Instances getInstances()
	{
		return dist.getInstances();
	}

	@Override
	public boolean getInvertSelection()
	{
		return dist.getInvertSelection();
	}

	@Override
	public String[] getOptions()
	{
		return dist.getOptions();
	}

	@Override
	public Enumeration<Option> listOptions()
	{
		return dist.listOptions();
	}

	@Override
	public void postProcessDistances(double[] distances)
	{
		dist.postProcessDistances(distances);
	}

	@Override
	public void setAttributeIndices(String value)
	{
		dist.setAttributeIndices(value);
	}

	@Override
	public void setInstances(Instances insts)
	{
		dist.setInstances(insts);
	}

	@Override
	public void setInvertSelection(boolean value)
	{
		dist.setInvertSelection(value);
	}

	@Override
	public void setOptions(String[] options) throws Exception
	{
		dist.setOptions(options);
	}

	@Override
	public void update(Instance ins)
	{
		dist.update(ins);
	}

	@Override
	public double distance(Instance first, Instance second, PerformanceStats stats) throws Exception
	{
		return dist.distance(first, second, stats);
	}

	@Override
	public double distance(Instance first, Instance second, double cutOffValue)
	{
		return dist.distance(first, second, cutOffValue);
	}

	@Override
	public double distance(Instance first, Instance second, double cutOffValue,
			PerformanceStats stats)
	{
		return dist.distance(first, second, cutOffValue, stats);
	}

	@Override
	public void clean()
	{
		dist.clean();
	}

}
