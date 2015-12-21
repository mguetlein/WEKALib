package org.mg.wekalib.eval2.job;

import org.mg.wekalib.eval2.data.DataSet;

public interface FeatureProvider extends JobOwner<DataSet[]>
{
	public void setTrainingDataset(DataSet train);

	public void setTestDataset(DataSet test);

	public DataSet getTrainingDataset();

	public String getName();
}
