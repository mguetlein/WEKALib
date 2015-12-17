package org.mg.wekalib.eval2;

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

	@Override
	public String toString()
	{
		return getName();
	}
}
