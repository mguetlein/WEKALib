package org.mg.wekalib.appdomain;

import org.mg.wekalib.distance.TanimotoDistance;

public class TanimotoNNADModel extends NNDistanceBasedADModel
{
	public TanimotoNNADModel(int k, boolean mean, double pThreshold)
	{
		super(k, mean, pThreshold, new TanimotoDistance(), true);
	}

}