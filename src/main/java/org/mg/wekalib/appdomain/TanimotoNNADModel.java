package org.mg.wekalib.appdomain;

import org.mg.wekalib.distance.TanimotoDistance;

public class TanimotoNNADModel extends NNDistanceBasedADModel
{
	private static final long serialVersionUID = 1L;

	public TanimotoNNADModel(int k, boolean mean, double pThreshold)
	{
		super(k, mean, pThreshold, new TanimotoDistance(), true);
	}

}