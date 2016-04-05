package org.mg.wekalib.appdomain;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.stat.inference.TestUtils;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.Binning;
import org.mg.javalib.util.DoubleArraySummary;
import org.mg.javalib.util.HashUtil;
import org.mg.javalib.util.SortedList;
import org.mg.javalib.util.SwingUtil;

import weka.core.Instance;
import weka.core.Instances;

public abstract class NNDistanceBasedAppDomainModel implements AppDomainModel
{
	// params

	int k = 3;

	double minP = 0.0001;

	// member variables

	double meanTrainingDistance;

	private Instances trainingData;

	protected abstract double computeDistance(Instance i1, Instance i2);

	protected abstract void buildInternal(Instances trainingData);

	private HashMap<Integer, Double> distances = new HashMap<>();

	public double getDistance(Instance i1, Instance i2)
	{
		Integer key = HashUtil.hashCode(i1, i2);
		if (!distances.containsKey(key))
		{
			double d = computeDistance(i1, i2);
			distances.put(key, d);
			distances.put(HashUtil.hashCode(i2, i1), d);
		}
		return distances.get(key);
	}

	@Override
	public void build(Instances trainingData)
	{
		buildInternal(trainingData);
		this.trainingData = trainingData;
		List<Double> knnTrainingDistances = new ArrayList<>();
		for (Instance instance : trainingData)
			knnTrainingDistances.add(computeKnnDist(instance));
		meanTrainingDistance = DoubleArraySummary.create(knnTrainingDistances).getMean();
		binning = new Binning(ArrayUtil.toPrimitiveDoubleArray(knnTrainingDistances), 10, false);
		SwingUtil.showInFrame(binning.plot());
	}

	Binning binning;

	private double computeKnnDist(Instance instance)
	{
		SortedList<Double> dists = new SortedList<>();
		for (Instance instance2 : trainingData)
		{
			if (instance == instance2)
				continue;
			double d = getDistance(instance, instance2);
			if (dists.size() < k)
				dists.add(d);
			else if (d < dists.get(k - 1))
			{
				dists.add(d);
				dists.remove(k);
			}
		}
		return dists.get(k - 1);
	}

	@Override
	public boolean isInsideAppdomain(Instance testInstance)
	{
		return pValue(testInstance) >= minP;
	}

	@Override
	public double pValue(Instance testInstance)
	{
		double dist = computeKnnDist(testInstance);
		if (dist <= meanTrainingDistance)
			return 1.0;
		long all[] = binning.getAllCounts();
		long selected[] = binning.getSelectedCounts(dist);
		double p = TestUtils.chiSquareTestDataSetsComparison(selected, all);
		System.out.println(p);
		if (p < minP)
		{
			System.out.println("distance: " + dist);
			SwingUtil.showInFrame(binning.plot(dist), dist + " : " + p, false);
		}
		return p;
	}

}
