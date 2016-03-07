package org.mg.wekalib.eval2;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;
import org.mg.javalib.util.CountedSet;
import org.mg.javalib.util.SetUtil;
import org.mg.wekalib.eval2.data.DataSet;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class FoldDatasetTest
{
	private double getTargetValueRatio(DataSet d)
	{
		CountedSet<String> s = CountedSet.create(d.getEndpoints());
		Assert.assertEquals(s.getNumValues(), 2);
		return s.getCount(s.values().get(1)) / (double) s.getCount(s.values().get(0));
	}

	@Test
	public void testFolds() throws FileNotFoundException, IOException
	{
		Instances inst = new Instances(
				new FileReader(System.getProperty("user.home") + "/data/weka/nominal/sonar.arff"));
		Attribute a = new Attribute("index");
		inst.insertAttributeAt(a, 0);
		for (int i = 0; i < inst.numInstances(); i++)
			inst.instance(i).setValue(inst.attribute(0), i);
		inst.setClassIndex(inst.numAttributes() - 1);
		WekaInstancesDataSet d = new WekaInstancesDataSet(inst, 1);
		double origTargetRatio = getTargetValueRatio(d);

		int numFolds = 5;
		long randomSeed = 1;

		for (boolean stratified : new boolean[] { true, false })
		{
			HashMap<Integer, Set<Integer>> testIdxPerFold = new HashMap<>();

			for (int repetition = 0; repetition < 2; repetition++)
			{
				double maxTestRatioDiff = 0;
				Set<Integer> joinedTestIdx = new HashSet<>();

				for (int fold = 0; fold < numFolds; fold++)
				{
					DataSet train = d.getTrainFold(numFolds, stratified, randomSeed, fold);
					DataSet test = d.getTestFold(numFolds, stratified, randomSeed, fold);
					// test dataset + train dataset sizes add up to entire dataset size 
					Assert.assertEquals(train.getSize() + test.getSize(), d.getSize());

					// endpoint-value-ratio in train is similar to ratio in entire
					Assert.assertEquals(origTargetRatio, getTargetValueRatio(train), stratified ? 0.01 : 0.1);
					double testRatio = getTargetValueRatio(test);
					// ratio in test is similar to entire ratio
					Assert.assertEquals(origTargetRatio, testRatio, stratified ? 0.075 : 0.5);
					maxTestRatioDiff = Math.max(maxTestRatioDiff, Math.abs(testRatio - origTargetRatio));

					Set<Integer> trainIdx = new HashSet<>();
					for (Instance instance : train.getWekaInstances())
						trainIdx.add((int) instance.value(train.getWekaInstances().attribute(0)));
					// train instances are uniq
					Assert.assertEquals(trainIdx.size(), train.getSize());

					Set<Integer> testIdx = new HashSet<>();
					for (Instance instance : test.getWekaInstances())
						testIdx.add((int) instance.value(test.getWekaInstances().attribute(0)));
					// test instances are uniq
					Assert.assertEquals(testIdx.size(), test.getSize());

					// train + test instances are disjunct
					Assert.assertEquals(0, SetUtil.intersectSize(trainIdx, testIdx));

					// test instances of this fold are disjunct to test instances in previous folds
					Assert.assertEquals(0, SetUtil.intersectSize(joinedTestIdx, testIdx));
					joinedTestIdx.addAll(testIdx);

					if (repetition == 0)
						testIdxPerFold.put(fold, testIdx);
					else // when repeated, test instances are equal
						Assert.assertEquals(testIdxPerFold.get(fold), testIdx);
				}

				System.out.println(maxTestRatioDiff);
				if (stratified) // max ratio diff is low when stratfied
					Assert.assertTrue(maxTestRatioDiff < 0.075);
				else // max ratio diff is higher when not stratified
					Assert.assertTrue(maxTestRatioDiff > 0.075);

				// disjunct test instances add up to entire dataset
				Assert.assertEquals(joinedTestIdx.size(), d.getSize());
			}
		}
	}

	public static void main(String[] args) throws FileNotFoundException, IOException
	{
		new FoldDatasetTest().testFolds();
	}
}
