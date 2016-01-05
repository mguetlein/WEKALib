package org.mg.wekalib.eval2.data;

import org.mg.wekalib.eval2.job.DefaultComposedKeyProvider;

public abstract class AbstractDataSet extends DefaultComposedKeyProvider implements DataSet
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

	@Override
	public String toString()
	{
		return getName();
	}

	@Override
	public String getKeyPrefix()
	{
		return getName();
	}
}
