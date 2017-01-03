package org.mg.wekalib.outlier;

import java.util.BitSet;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class SOD extends AbstractClassifier
{
	private static final long serialVersionUID = 1L;

	DistanceFunction dist = new EuclideanDistance();
	int knn = 10;
	double alpha = 0.8;
	NearestNeighbourSearch nnSearch = new LinearNNSearch();

	Instances train;

	public SOD()
	{
	}

	@Override
	public double classifyInstance(Instance instance) throws Exception
	{
		Instances nns = nnSearch.kNearestNeighbours(instance, knn);
		double sod;
		if (nns.size() > 0)
		{
			Instance center = centroid(nns);

			// Note: per-dimension variances; no covariances.
			double[] variances = variances(nns);
			double expectationOfVariance = new DescriptiveStatistics(variances).getMean();
			BitSet attrFilter = new BitSet(variances.length);
			for (int d = 0; d < variances.length; d++)
				if (variances[d] < alpha * expectationOfVariance)
					attrFilter.set(d);

			sod = subspaceOutlierDegree(instance, center, attrFilter);
		}
		else
			sod = 0;
		return sod;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception
	{
		train = data;
		dist.setInstances(data);
		nnSearch.setDistanceFunction(dist);
	}

	private double subspaceOutlierDegree(Instance inst, Instance center, BitSet attrFilter)
	{
		final int card = attrFilter.cardinality();
		if (card == 0)
			return 0;

		throw new NotImplementedException("unclear: what to do with normalization");
		//		final SubspaceEuclideanDistanceFunction df = new SubspaceEuclideanDistanceFunction(
		//				weightVector);
		//		double distance = df.distance(queryObject, center);
		//		distance /= card; // FIXME: defined and published as card, should be
		//							// sqrt(card), unfortunately
		//		return distance;
	}

	protected double[] variances(Instances inst)
	{
		double var[] = new double[inst.numAttributes()];
		for (int i = 0; i < inst.numAttributes(); i++)
			var[i] = inst.variance(i);
		return var;
	}

	protected Instance centroid(Instances inst)
	{
		double[] vals = new double[inst.numAttributes()];
		for (int j = 0; j < inst.numAttributes(); j++)
		{
			if (dist instanceof EuclideanDistance)
				vals[j] = inst.meanOrMode(j);
			else
				throw new IllegalStateException();
		}
		return new DenseInstance(1.0, vals);
	}

}
