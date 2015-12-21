package org.mg.wekalib.eval2.model;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.JobOwner;
import org.mg.wekautil.Predictions;

public interface Model extends JobOwner<Predictions> //, Serializable
{
	//	public void setFeatureProvider(FeatureProvider p);

	public void setTrainingDataset(DataSet train);

	public void setTestDataset(DataSet test);

	public String getName();

	//	public DataSet getTrainingDataset();

	//	public FeatureProvider getFeatureProvider();
}
