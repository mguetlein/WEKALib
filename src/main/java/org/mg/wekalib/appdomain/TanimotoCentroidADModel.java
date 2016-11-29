package org.mg.wekalib.appdomain;

import java.util.BitSet;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.mg.wekalib.distance.TanimotoDistance;

import weka.core.Instance;
import weka.core.SparseInstance;

public class TanimotoCentroidADModel extends CentroidDistanceBasedADModel
{
	double consensusFPThreshold;

	public TanimotoCentroidADModel(double pThreshold, double consensusFPThreshold)
	{
		super(pThreshold, new TanimotoDistance(), true);
		this.consensusFPThreshold = consensusFPThreshold;
	}

	@Override
	protected Instance createCentroid()
	{
		int counts[] = new int[trainingData.numAttributes()];

		DescriptiveStatistics stats = new DescriptiveStatistics();
		for (Instance i : trainingData)
		{
			BitSet bs = ((TanimotoDistance) dist).getBitSet(i);
			stats.addValue(bs.cardinality());
			for (int j = 0; j < counts.length; j++)
				if (bs.get(j))
					counts[j]++;
		}

		BitSet centroid = new BitSet();
		for (int j = 0; j < counts.length; j++)
			if (counts[j] >= (trainingData.size() * consensusFPThreshold))
				centroid.set(j);

		System.err.println("num features: " + (trainingData.numAttributes() - 1));
		System.err.println(
				"training cardinality: " + stats.getMean() + " +- " + stats.getStandardDeviation());
		System.err.println("centroid cardinality: " + centroid.cardinality());

		Instance centroidInstance = new SparseInstance(trainingData.numAttributes());
		centroidInstance.setDataset(trainingData);
		for (int j = 0; j < counts.length; j++)
			if (centroid.get(j))
				centroidInstance.setValue(j, 1.0);

		return centroidInstance;
	}

}