package org.mg.wekalib.eval2.data;

import java.util.List;

import org.mg.wekalib.eval2.job.FeatureProvider;
import org.mg.wekalib.eval2.job.Printer;

import weka.core.Instances;

public class FeatureProvidedDataSet extends AbstractDataSet implements WrappedDataSet
{
	protected FeatureProvider featureProvider;
	protected boolean train;

	private DataSet self;
	private Instances instances;

	public FeatureProvidedDataSet(FeatureProvider featureProvider, boolean train)
	{
		this.featureProvider = featureProvider;
		this.train = train;
	}

	public FeatureProvider getFeatureProvider()
	{
		return featureProvider;
	}

	@Override
	public int getPositiveClass()
	{
		return featureProvider.getTrainingDataset().getPositiveClass();
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(featureProvider, train);
	}

	@Override
	public DataSet getFilteredDataset(String name, List<Integer> idx)
	{
		return getSelf().getFilteredDataset(name, idx);
	}

	@Override
	public int getSize()
	{
		return getSelf().getSize();
	}

	@Override
	public List<String> getEndpoints()
	{
		return getSelf().getEndpoints();
	}

	public DataSet getSelf()
	{
		if (self == null)
		{
			DataSet res[] = featureProvider.getResult();
			if (res == null)
			{
				Printer.println("no result! " + featureProvider.getTrainingDataset());
				System.exit(1);
			}
			self = res[train ? 0 : 1];
		}
		return self;
	}

	public String getName()
	{
		// name is used as key prefix, keep it simple		
		//		return parent.getName() + "/" + (train ? "Train" : "Test") + "-fold-" + (fold + 1) + "-of-" + numFolds;
		return featureProvider.getTrainingDataset().getName();
	}

	@Override
	public Instances getWekaInstances()
	{
		if (instances == null)
			instances = getSelf().getWekaInstances();
		return instances;
	}

}
