package org.mg.wekalib.eval2.data;

import java.util.List;

import org.mg.wekalib.eval2.job.ComposedKeyProvider;
import org.mg.wekalib.eval2.job.KeyProvider;

import weka.core.Instances;

public interface DataSet extends KeyProvider, ComposedKeyProvider
{
	public DataSet getTrainFold(int numFolds, long randomSeed, int fold);

	public DataSet getTestFold(int numFolds, long randomSeed, int fold);

	public Instances getWekaInstances();

	public int getPositiveClass();

	public DataSet getFilteredDataset(String name, List<Integer> idx);

	public int getSize();

	public String getName();
}
