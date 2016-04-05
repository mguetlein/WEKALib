package org.mg.wekalib.appdomain;

import weka.core.Instance;
import weka.core.Instances;

public interface AppDomainModel
{
	public void build(Instances trainingData);

	public boolean isInsideAppdomain(Instance testInstance);

	public double pValue(Instance testInstance);
}
