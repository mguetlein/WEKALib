package org.mg.wekalib.appdomain;

import org.mg.javalib.util.DoubleArraySummary;
import org.mg.javalib.util.SortedList;
import org.mg.wekalib.distance.Distance;

import weka.core.Instance;

public class NNDistanceBasedADModel extends AbstractDistanceBasedADModel
{
	protected int k;

	protected boolean mean;

	public NNDistanceBasedADModel(int k, boolean mean, double pThreshold, Distance dist,
			boolean fitToNormal)
	{
		super(pThreshold, fitToNormal, dist);
		this.k = k;
		this.mean = mean;
	}

	@Override
	protected double computeDistanceInternal(int trainingIdx)
	{
		return computeDistanceToTraining(null, trainingIdx);
	}

	@Override
	public double computeDistanceInternal(Instance testInstance)
	{
		return computeDistanceToTraining(testInstance, -1);
	}

	protected double computeDistanceToTraining(Instance testInstance, int trainingIdx)
	{
		SortedList<Double> dists = new SortedList<>();
		for (int i = 0; i < trainingData.numInstances(); i++)
		{
			if (trainingIdx != -1 && trainingIdx == i)
				continue;

			double d;
			if (trainingIdx == -1)
				d = dist.distance(testInstance, i);
			else
				d = dist.distance(trainingIdx, i);
			if (dists.size() < k)
			{
				dists.add(d);
			}
			else if (d < dists.get(k - 1))
			{
				dists.add(d);
				dists.remove(k);
			}
		}
		//			System.out.println(dists);
		Double d;
		if (mean)
			d = DoubleArraySummary.create(dists).getMean();
		else
			d = DoubleArraySummary.create(dists).getMedian();
		//			System.out.println("#cached dist values " + cachedDist.size());
		return d;
	}

}
