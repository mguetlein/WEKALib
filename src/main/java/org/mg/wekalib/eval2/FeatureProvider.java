package org.mg.wekalib.eval2;

public interface FeatureProvider extends JobOwner<DataSet[]>
{

	public void setTrainingDataset(DataSet train);

	public void setTestDataset(DataSet test);

	public FeatureProvider cloneFeatureProvider();

	public DataSet getTrainingDataset();

	public String getName();

}
