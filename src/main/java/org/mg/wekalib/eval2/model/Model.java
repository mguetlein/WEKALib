package org.mg.wekalib.eval2.model;

import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.job.JobOwner;
import org.mg.wekalib.evaluation.Predictions;

public interface Model extends JobOwner<Predictions> //, Serializable
{
	//	public void setFeatureProvider(FeatureProvider p);

	public void setTrainingDataset(DataSet train);

	public void setTestDataset(DataSet test);

	public String getName();

	public DataSet getTrainingDataset();

	public String getAlgorithmShortName();

	public String getAlgorithmParamsNice();

	public boolean isValid(DataSet dataSet);

	//	public FeatureProvider getFeatureProvider();
}
