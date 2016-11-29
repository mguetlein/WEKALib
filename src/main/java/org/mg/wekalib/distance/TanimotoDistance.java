package org.mg.wekalib.distance;

import java.io.Serializable;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import org.mg.javalib.util.BitSetUtil;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.neighboursearch.PerformanceStats;

public class TanimotoDistance implements Distance, DistanceFunction, Serializable
{
	private static final long serialVersionUID = 1L;

	public static final long VERSION = 1;

	private static boolean VERBOSE = true;

	Instances train;

	Map<Instance, BitSet> bitsets = new HashMap<>();

	int lookup = 0;

	public BitSet getBitSet(Instance inst)
	{
		//		if (!bitsets.containsKey(inst))
		//		{
		//			if (VERBOSE && bitsets.size() > 0 && bitsets.size() % 100 == 0)
		//				System.out.println("Tanimoto> storing bitset for compound #" + bitsets.size()
		//						+ ", lookups: " + lookup);
		//
		BitSet bs = new BitSet();
		for (int i = 0; i < inst.numAttributes(); i++)
		{
			if (inst.classIndex() == i)
				continue;
			if (inst.value(i) == 1.0)
				bs.set(i);
		}
		return bs;
		//			bitsets.put(inst, bs);
		//		}
		//		else
		//			lookup++;
		//		return bitsets.get(inst);
	}

	public String toString()
	{
		return this.getClass().toString() + "#" + VERSION;
	}

	public static TanimotoDistance fromString(String s)
	{
		if (s.startsWith(TanimotoDistance.class.toString()) && s.endsWith(VERSION + ""))
			return new TanimotoDistance();
		else
			return null;
	}

	@Override
	public void build(Instances train)
	{
		if (VERBOSE)
			System.out.println("Tanimoto> training data has " + train.numInstances()
					+ " instances and " + train.numAttributes() + " attributes");
		for (int i = 0; i < train.numAttributes(); i++)
		{
			if (train.classIndex() == i)
				continue;
			if (!train.attribute(i).isNominal())
				throw new IllegalArgumentException("not nominal : " + train.attribute(i));
			if (train.attribute(i).numValues() != 2)
				throw new IllegalArgumentException("not binary : " + train.attribute(i));
		}
		this.train = train;
	}

	@Override
	public double distance(int trainInstanceIdx1, int trainInstanceIdx2)
	{
		return distanceInternal(train.get(trainInstanceIdx1), train.get(trainInstanceIdx2));
	}

	@Override
	public double distance(Instance testInstance, int trainInstanceIdx)
	{
		return distanceInternal(testInstance, train.get(trainInstanceIdx));
	}

	@Override
	public double distance(Instance testInstance1, Instance testInstance2)
	{
		return distanceInternal(testInstance1, testInstance2);
	}

	public double distanceInternal(Instance i1, Instance i2)
	{
		double d = 1 - BitSetUtil.tanimotoSimilarity(getBitSet(i1), getBitSet(i2));
		//		System.out.println(d);
		return d;
	}

	@Override
	public Double getMaxDistance()
	{
		return 1.0;
	}

	// weka methods

	@Override
	public void clean()
	{
		bitsets.clear();
	}

	@Override
	public double distance(Instance first, Instance second, double cutOffValue)
	{
		return distance(first, second, cutOffValue, null);
	}

	@Override
	public double distance(Instance first, Instance second, PerformanceStats stats) throws Exception
	{
		return distance(first, second, Double.POSITIVE_INFINITY, stats);
	}

	public static double tanimotoDistanceDirect(Instance i1, Instance i2)
	{
		//		StopWatchUtil.start("tani");
		int and = 0;
		int or = 0;
		for (int i = 0; i < i1.numAttributes(); i++)
		{
			if (i1.classIndex() == i)
				continue;
			if (i1.value(i) == 1.0)
			{
				if (i2.value(i) == 1.0)
					and++;
				else
					or++;
			}
			else if (i2.value(i) == 1.0)
				or++;
		}
		if (or == 0)
			return 0;
		double d = 1 - and / (double) or;
		//		StopWatchUtil.stop("tani");
		return d;
	}

	@Override
	public double distance(Instance first, Instance second, double cutOffValue,
			PerformanceStats stats)
	{
		//double distance = distanceInternal(first, second);
		double distance = tanimotoDistanceDirect(first, second);
		if (stats != null)
			stats.incrCoordCount();
		if (distance > cutOffValue)
			return Double.POSITIVE_INFINITY;
		return distance;
	}

	@Override
	public String getAttributeIndices()
	{
		return "all";
	}

	@Override
	public Instances getInstances()
	{
		return train;
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
		bitsets.remove(ins);
	}

}
