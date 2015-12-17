package org.mg.wekalib.eval2;

import java.io.Serializable;

import org.mg.wekautil.Predictions;

public interface Model extends JobOwner<Predictions>, Serializable
{
	public Model cloneModel();

	//	public void setFeatureProvider(FeatureProvider p);

	public void setTrainingDataset(DataSet train);

	public void setTestDataset(DataSet test);

	public String getName();

	//	public DataSet getTrainingDataset();

	//	public FeatureProvider getFeatureProvider();
}
