package org.mg.wekalib.appdomain;

import weka.core.Instance;

public interface DistanceBasedADModel extends AppDomainModel
{
	public double computeDistance(Instance instance);

	public double getMaxTrainingDistance();
}
