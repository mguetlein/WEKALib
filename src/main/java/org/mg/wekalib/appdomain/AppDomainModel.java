package org.mg.wekalib.appdomain;

import java.io.Serializable;

import weka.core.Instance;
import weka.core.Instances;

public interface AppDomainModel extends Serializable
{
	public void build(Instances trainingData);

	public boolean isInsideAppdomain(Instance testInstance);
}
