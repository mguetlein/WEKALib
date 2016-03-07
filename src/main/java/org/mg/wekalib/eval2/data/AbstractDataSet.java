package org.mg.wekalib.eval2.data;

import org.mg.wekalib.eval2.job.DefaultComposedKeyProvider;

public abstract class AbstractDataSet extends DefaultComposedKeyProvider implements DataSet
{
	@Override
	public DataSet getTrainFold(int numFolds, boolean stratified, long randomSeed, int fold)
	{
		return new FoldDataSet(this, numFolds, stratified, randomSeed, fold, true);
	}

	@Override
	public DataSet getTestFold(int numFolds, boolean stratified, long randomSeed, int fold)
	{
		return new FoldDataSet(this, numFolds, stratified, randomSeed, fold, false);
	}

	@Override
	public DataSet getTrainSplit(double ratio, boolean stratified, long randomSeed)
	{
		return new SplitDataSet(this, ratio, stratified, randomSeed, true);
	}

	@Override
	public DataSet getTestSplit(double ratio, boolean stratified, long randomSeed)
	{
		return new SplitDataSet(this, ratio, stratified, randomSeed, false);
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
