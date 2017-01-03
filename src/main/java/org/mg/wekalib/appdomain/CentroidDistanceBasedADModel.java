package org.mg.wekalib.appdomain;

import org.mg.wekalib.distance.Distance;

import weka.core.Instance;

public abstract class CentroidDistanceBasedADModel extends AbstractDistanceBasedADModel
{
	private static final long serialVersionUID = 1L;

	private Instance centroid;

	public CentroidDistanceBasedADModel(double pThreshold, Distance dist, boolean fitToNormal)
	{
		super(pThreshold, fitToNormal, dist);
	}

	protected abstract Instance createCentroid();

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
		if (centroid == null)
			centroid = createCentroid();
		double d;
		if (trainingIdx == -1)
			d = dist.distance(centroid, testInstance);
		else
			d = dist.distance(centroid, trainingIdx);

		return d;
	}

}
