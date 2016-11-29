package org.mg.wekalib.appdomain;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.inference.TestUtils;
import org.mg.javalib.freechart.HistogramPanel;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.SwingUtil;

import weka.core.Instance;
import weka.core.Instances;

public abstract class DistanceDistributionBasedAppDomainModel implements AppDomainModel
{
	private Instances trainingData;

	double[] trainingDistances;

	protected abstract double computeDistance(Instance i1, Instance i2);

	protected abstract void buildInternal(Instances trainingData);

	@Override
	public void build(Instances trainingData)
	{
		buildInternal(trainingData);
		this.trainingData = trainingData;
		List<Double> trainingDist = new ArrayList<>();
		for (int i = 0; i < trainingData.size() - 1; i++)
			for (int j = i + 1; j < trainingData.size(); j++)
				trainingDist.add(computeDistance(trainingData.get(i), trainingData.get(j)));
		trainingDistances = ArrayUtil.toPrimitiveDoubleArray(trainingDist);
	}

	@Override
	public boolean isInsideAppdomain(Instance testInstance)
	{
		return true;
	}

	@Override
	public double pValue(Instance testInstance)
	{
		double[] testDistances = new double[trainingData.size()];
		for (int i = 0; i < trainingData.size(); i++)
			testDistances[i] = computeDistance(trainingData.get(i), testInstance);
		List<double[]> testVals = new ArrayList<>();
		testVals.add(trainingDistances);
		testVals.add(testDistances);
		double p = TestUtils.kolmogorovSmirnovStatistic(trainingDistances, testDistances);

		//		double dist = computeKnnDist(testInstance);
		//		long all[] = binning.getAllCounts();
		//		long selected[] = binning.getSelectedCounts(dist);
		//		double p = TestUtils.chiSquareTestDataSetsComparison(selected, all);
		System.out.println(p);
		if (p < 0.05)
		{
			HistogramPanel panel = new HistogramPanel("Binning", null, "x", "y",
					ListUtil.createList("training", "test"), testVals, 25);
			SwingUtil.showInFrame(panel);
		}
		return p;
	}
}
