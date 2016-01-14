package org.mg.wekalib.eval2.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mg.javalib.util.ArrayUtil;
import org.mg.wekalib.eval2.job.Printer;

import weka.core.Instances;

public class FoldDataSet extends AbstractDataSet
{
	protected DataSet parent;
	protected int numFolds;
	protected long randomSeed;
	protected int fold;
	protected boolean train;

	private DataSet self;
	private Instances instances;

	public FoldDataSet(DataSet parent, int numFolds, long randomSeed, int fold, boolean train)
	{
		this.parent = parent;
		this.numFolds = numFolds;
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

	public DataSet getSelf()
	{
		if (self == null)
		{
			Printer.println("FoldDataset: creating " + (train ? "train" : "test") + " fold "
					+ (fold + 1) + "/" + numFolds + ", seed " + randomSeed);
			Integer[] idx = ArrayUtil.toIntegerArray(ArrayUtil.indexArray(parent.getSize()));
			ArrayUtil.scramble(idx, new Random(randomSeed));
			List<Integer[]> cvIdx = ArrayUtil.split(idx, numFolds);
			List<Integer> selfIdx = new ArrayList<>();
			for (int f = 0; f < cvIdx.size(); f++)
				if ((train && fold != f) || (!train && fold == f))
					selfIdx.addAll(ArrayUtil.toList(cvIdx.get(f)));
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
