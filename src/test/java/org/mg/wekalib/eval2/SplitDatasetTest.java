package org.mg.wekalib.eval2;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;
import org.mg.javalib.util.CountedSet;
import org.mg.javalib.util.SetUtil;
import org.mg.javalib.util.StringUtil;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class SplitDatasetTest
{
	private double getTargetValueRatio(DataSet d)
	{
		CountedSet<String> s = CountedSet.create(d.getEndpoints());
		Assert.assertEquals(s.getNumValues(), 2);
		return s.getCount(s.values().get(1)) / (double) s.getCount(s.values().get(0));
	}

	@Test
	public void testSplits() throws FileNotFoundException, IOException
	{
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/nominal/breast-w.arff"));
		Attribute a = new Attribute("index");
		inst.insertAttributeAt(a, 0);
		for (int i = 0; i < inst.numInstances(); i++)
			inst.instance(i).setValue(inst.attribute(0), i);
		inst.setClassIndex(inst.numAttributes() - 1);
		WekaInstancesDataSet d = new WekaInstancesDataSet(inst, 1);
		double origTargetRatio = getTargetValueRatio(d);

		double splitRatio = 0.8;
		long randomSeed = 2;

		for (boolean stratified : new boolean[] { true, false })
		{
			System.out.println("\nstratified " + stratified);
			Set<Integer> testIdxOld = new HashSet<>();

			for (int repetition = 0; repetition < 2; repetition++)
			{
				DataSet train = d.getTrainSplit(splitRatio, stratified, randomSeed);
				DataSet test = d.getTestSplit(splitRatio, stratified, randomSeed);
				System.out.println(train.getSize() + " " + test.getSize() + " " + d.getSize());
				// test dataset + train dataset sizes add up to entire dataset size 
				Assert.assertEquals(train.getSize() + test.getSize(), d.getSize());

				System.out.println(StringUtil.formatDouble(getTargetValueRatio(train)) + " "
						+ StringUtil.formatDouble(getTargetValueRatio(test)) + " "
						+ StringUtil.formatDouble(origTargetRatio));
				// endpoint-value-ratio in train is similar to ratio in entire
				Assert.assertEquals(origTargetRatio, getTargetValueRatio(train),
						stratified ? 0.01 : 0.1);
				// ratio in test is similar to entire ratio
				Assert.assertEquals(origTargetRatio, getTargetValueRatio(test),
						stratified ? 0.075 : 0.5);

				Set<Integer> trainIdx = new HashSet<>();
				for (Instance instance : train.getWekaInstances())
					trainIdx.add((int) instance.value(train.getWekaInstances().attribute(0)));
				System.out.println(trainIdx);
				// train instances are uniq
				Assert.assertEquals(trainIdx.size(), train.getSize());

				Set<Integer> testIdx = new HashSet<>();
				for (Instance instance : test.getWekaInstances())
					testIdx.add((int) instance.value(test.getWekaInstances().attribute(0)));
				System.out.println(testIdx);
				// test instances are uniq
				Assert.assertEquals(testIdx.size(), test.getSize());

				// train + test instances are disjunct
				Assert.assertEquals(0, SetUtil.intersectSize(trainIdx, testIdx));

				if (repetition == 0)
					testIdxOld = testIdx;
				else // when repeated, test instances are equal
					Assert.assertEquals(testIdxOld, testIdx);
			}
		}
	}

	public static void main(String[] args) throws FileNotFoundException, IOException
	{
		new SplitDatasetTest().testSplits();
	}
}
