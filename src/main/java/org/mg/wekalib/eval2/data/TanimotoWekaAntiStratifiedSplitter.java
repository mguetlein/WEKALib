package org.mg.wekalib.eval2.data;

import org.mg.wekalib.distance.Distance;
import org.mg.wekalib.distance.TanimotoDistance;

public class TanimotoWekaAntiStratifiedSplitter extends WekaAntiStratifiedSplitter
{
	@Override
	public Distance getDistance()
	{
		return new TanimotoDistance();
	}
}
