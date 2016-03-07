package org.mg.wekalib.eval2.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.wekalib.eval2.job.Printer;

import weka.core.Instances;

public class SplitDataSet extends AbstractDataSet implements WrappedDataSet
{
	protected DataSet parent;
	protected double splitRatio;
	protected boolean stratified;
	protected long randomSeed;
	protected boolean train;

	private DataSet self;
	private Instances instances;

	public SplitDataSet(DataSet parent, double splitRatio, boolean stratified, long randomSeed,
			boolean train)
	{
		this.parent = parent;
		this.splitRatio = splitRatio;
		this.stratified = stratified;
		this.randomSeed = randomSeed;
		this.train = train;
	}

	@Override
	public int getPositiveClass()
	{
		return parent.getPositiveClass();
	}

	@Override
	public String getKeyContent()
	{
		return getKeyContent(parent, splitRatio, randomSeed, train, stratified);
	}

	public double getSplitRatio()
	{
		return splitRatio;
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
			Printer.println(
					"SplitDataset: creating " + (train ? "train" : "test") + " split, ratio "
							+ splitRatio + ", seed " + randomSeed + ", stratified " + stratified);
			List<Integer> selfIdx = new ArrayList<>();
			if (stratified)
			{
				Random r = new Random(randomSeed);
				HashMap<String, List<Integer>> clazzIndices = new HashMap<>();
				int i = 0;
				for (String clazz : parent.getEndpoints())
				{
					if (!clazzIndices.containsKey(clazz))
						clazzIndices.put(clazz, new ArrayList<Integer>());
					clazzIndices.get(clazz).add(i++);
				}
				int prevSize = 0;
				String s = "";
				for (String clazz : clazzIndices.keySet())
				{
					Integer[] idx = ArrayUtil.toIntegerArray(clazzIndices.get(clazz));
					ArrayUtil.scramble(idx, r);
					if (train)
						for (int j = 0; j < (int) (idx.length * splitRatio); j++)
							selfIdx.add(idx[j]);
					else
						for (int j = (int) (idx.length * splitRatio); j < idx.length; j++)
							selfIdx.add(idx[j]);
					s += (selfIdx.size() - prevSize) + " x " + clazz + " ";
					prevSize = selfIdx.size();
				}
				Printer.println("-> " + s);
				ListUtil.scramble(selfIdx, new Random(randomSeed));
			}
			else
			{
				Integer[] idx = ArrayUtil.toIntegerArray(ArrayUtil.indexArray(parent.getSize()));
				ArrayUtil.scramble(idx, new Random(randomSeed));
				if (train)
					for (int i = 0; i < (int) (idx.length * splitRatio); i++)
						selfIdx.add(idx[i]);
				else
					for (int i = (int) (idx.length * splitRatio); i < idx.length; i++)
						selfIdx.add(idx[i]);
			}
			self = parent.getFilteredDataset(getName(), selfIdx);
		}
		return self;
	}

	public String getName()
	{
		// name is used as key prefix, keep it simple		
		//		return parent.getName() + "/" + (train ? "Train" : "Test") + "-fold-" + (fold + 1) + "-of-" + numFolds;
		return parent.getName();
	}

	@Override
	public Instances getWekaInstances()
	{
		if (instances == null)
			instances = getSelf().getWekaInstances();
		return instances;
	}

}
