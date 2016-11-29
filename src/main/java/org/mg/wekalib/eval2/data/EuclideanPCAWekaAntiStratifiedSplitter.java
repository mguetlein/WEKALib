package org.mg.wekalib.eval2.data;

import org.mg.wekalib.distance.Distance;
import org.mg.wekalib.distance.EuclideanPCADistance;

public class EuclideanPCAWekaAntiStratifiedSplitter extends WekaAntiStratifiedSplitter
{
	@Override
	public Distance getDistance()
	{
		return new EuclideanPCADistance();
	}
}
