package org.mg.wekalib.appdomain;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.EmpiricalDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.core.Instances;

public class AppDomainFilter
{
	AppDomainModel model;

	public AppDomainFilter(AppDomainModel model)
	{
		this.model = model;
	}

	public void build(Instances trainingData)
	{
		model.build(trainingData);
	}

	public Instances apply(Instances testData)
	{
		Instances inst = new Instances(testData);
		for (int i = testData.size() - 1; i >= 0; i--)
			if (!model.isInsideAppdomain(testData.get(i)))
				inst.remove(i);
		System.out.println(inst.size() + "/" + testData.size() + " is inside app-domain");
		return inst;
	}

	public static void main(String[] args) throws FileNotFoundException, IOException
	{
		//		Instances train = new Instances(new FileReader("/tmp/train.arff"));
		//		Instances test = new Instances(new FileReader("/tmp/test.arff"));
		//		AppDomainFilter filter = new AppDomainFilter(new TanimotoNNADModel(3, true, 0.95));
		//		filter.build(train);
		//		filter.apply(test);

		double values[] = new NormalDistribution(0, 1).sample(100);

		DescriptiveStatistics stats = new DescriptiveStatistics(values);
		System.out.println(stats.getPercentile(0.00001));
		System.out.println(stats.getPercentile(50));
		System.out.println(stats.getPercentile(100));
		System.out.println();

		EmpiricalDistribution distribution = new EmpiricalDistribution(values.length / 10);
		distribution.load(values);

		System.out.println(distribution.cumulativeProbability(-30));
		System.out.println(distribution.cumulativeProbability(-3));
		System.out.println(distribution.cumulativeProbability(-2));
		System.out.println(distribution.cumulativeProbability(-1));
		System.out.println(distribution.cumulativeProbability(-0.5));
		System.out.println(distribution.cumulativeProbability(0));
		System.out.println(distribution.cumulativeProbability(0.5));
		System.out.println(distribution.cumulativeProbability(1));
		System.out.println(distribution.cumulativeProbability(2));
		System.out.println(distribution.cumulativeProbability(3));
		System.out.println(distribution.cumulativeProbability(30));

	}
}
