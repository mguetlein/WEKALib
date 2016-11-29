package org.mg.wekalib.distance;

public class DistanceSerializer
{
	public static Distance fromString(String s)
	{
		if (TanimotoDistance.fromString(s) != null)
			return TanimotoDistance.fromString(s);
		if (EuclideanPCADistance.fromString(s) != null)
			return EuclideanPCADistance.fromString(s);
		throw new IllegalArgumentException(s);
	}
}
