package org.mg.wekalib.appdomain;

import org.mg.wekalib.distance.EuclideanPCADistance;

public class PCAEuclideanADModel extends NNDistanceBasedADModel
{
	private static final long serialVersionUID = 1L;

	public PCAEuclideanADModel(int k, boolean mean, double pValueThreshold)
	{
		super(k, mean, pValueThreshold, new EuclideanPCADistance(), false);
	}

}
