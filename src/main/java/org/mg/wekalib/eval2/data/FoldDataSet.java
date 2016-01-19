package org.mg.wekalib.eval2.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.wekalib.eval2.job.Printer;

import weka.core.Instances;

public class FoldDataSet extends AbstractDataSet
{
	protected DataSet parent;
	protected int numFolds;
	protected boolean stratified;
	protected long randomSeed;
	protected int fold;
	protected boolean train;

	private DataSet self;
	private Instances instances;

	public FoldDataSet(DataSet parent, int numFolds, boolean stratified, long randomSeed, int fold,
			boolean train)
	{
		this.parent = parent;
		this.numFolds = numFolds;
		this.stratified = stratified;
		this.randomSeed = randomSeed;
		this.fold = fold;
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
		if (stratified)
			return getKeyContent(parent, numFolds, randomSeed, fold, train, stratified);
		else
			return getKeyContent(parent, numFolds, randomSeed, fold, train);
	}

	public int getFold()
	{
		return fold;
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
			Printer.println("FoldDataset: creating " + (train ? "train" : "test") + " fold "
					+ (fold + 1) + "/" + numFolds + ", seed " + randomSeed + ", stratified "
					+ stratified);
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
				for (String clazz : clazzIndices.keySet())
				{
					Integer[] idx = ArrayUtil.toIntegerArray(clazzIndices.get(clazz));
					ArrayUtil.scramble(idx, r);
					List<Integer[]> cvIdx = ArrayUtil.split(idx, numFolds);
					for (int f = 0; f < cvIdx.size(); f++)
						if ((train && fold != f) || (!train && fold == f))
							selfIdx.addAll(ArrayUtil.toList(cvIdx.get(f)));
				}
				ListUtil.scramble(selfIdx);
			}
			else
			{
				Integer[] idx = ArrayUtil.toIntegerArray(ArrayUtil.indexArray(parent.getSize()));
				ArrayUtil.scramble(idx, new Random(randomSeed));
				List<Integer[]> cvIdx = ArrayUtil.split(idx, numFolds);
				for (int f = 0; f < cvIdx.size(); f++)
					if ((train && fold != f) || (!train && fold == f))
						selfIdx.addAll(ArrayUtil.toList(cvIdx.get(f)));
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
