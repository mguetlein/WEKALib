package org.mg.wekalib.eval2.data;

import java.util.List;

import org.mg.wekalib.eval2.job.ComposedKeyProvider;
import org.mg.wekalib.eval2.job.KeyProvider;

import weka.core.Instances;

public interface DataSet extends KeyProvider, ComposedKeyProvider
{
	public DataSet getTrainFold(int numFolds, boolean stratified, long randomSeed, int fold);

	public DataSet getTestFold(int numFolds, boolean stratified, long randomSeed, int fold);

	public DataSet getTrainSplit(double ratio, boolean stratified, long randomSeed,
			AntiStratifiedSplitter antiStratifiedSplitter);

	public DataSet getTestSplit(double ratio, boolean stratified, long randomSeed,
			AntiStratifiedSplitter antiStratifiedSplitter);

	public Instances getWekaInstances();

	public int getPositiveClass();

	public DataSet getFilteredDataset(String name, List<Integer> idx);

	public int getSize();

	public String getName();

	public List<String> getEndpoints();
}
