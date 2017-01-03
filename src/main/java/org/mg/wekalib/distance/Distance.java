package org.mg.wekalib.distance;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;

public interface Distance extends Serializable
{
	public void build(Instances train);

	public double distance(int trainInstanceIdx1, int trainInstanceIdx2);

	public double distance(Instance testInstance, int trainInstanceIdx);

	public double distance(Instance testInstance1, Instance testInstance2);

	public Double getMaxDistance();
}
