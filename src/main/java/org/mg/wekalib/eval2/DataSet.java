package org.mg.wekalib.eval2;

import java.util.List;

import weka.core.Instances;

public interface DataSet
{
	public DataSet getTrainFold(int numFolds, long randomSeed, int fold);

	public DataSet getTestFold(int numFolds, long randomSeed, int fold);

	//	public DataSet cloneDataset();

	public Instances getWekaInstances();

	public DataSet getFilteredDataset(String name, List<Integer> idx);

	public int getSize();

	public String getName();

}
