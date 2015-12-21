package org.mg.wekalib.eval2.data;

import org.mg.wekalib.eval2.job.DefaultJobOwner;

public abstract class AbstractDataSet implements DataSet
{
	@Override
	public DataSet getTrainFold(int numFolds, long randomSeed, int fold)
	{
		return new FoldDataSet(this, numFolds, randomSeed, fold, true);
	}

	@Override
	public DataSet getTestFold(int numFolds, long randomSeed, int fold)
	{
		return new FoldDataSet(this, numFolds, randomSeed, fold, false);
	}

	public String getKey(Object... object)
	{
		return DefaultJobOwner.getKey(this.getClass(), object);
	}

	@Override
	public String toString()
	{
		return getName();
	}
}
