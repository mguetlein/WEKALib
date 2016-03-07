package org.mg.wekalib.eval2;

import java.io.FileReader;
import java.util.HashMap;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.Predictions;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class HoldoutTest
{
	@Test
	public void runHoldoutandCompareWithWekaHoldout()
	{
		try
		{
			HashMap<PredictionUtil.ClassificationMeasure, Double> valP = new HashMap<>();
			HashMap<PredictionUtil.ClassificationMeasure, Double> valE = new HashMap<>();
			int run = 0;
			double maxDiff = 0;

			// run repeated Holdouts on the same data until the max difference
			// between Holdout() and Weka-Holdout
			// in mean statistics for each value is below 0.005
			do
			{
				Holdout ho = new Holdout();
				ho.setRandomSeed(run);
				ho.setStratified(false);
				ho.setSplitRatio(0.66);
				Instances inst = new Instances(new FileReader(
						System.getProperty("user.home") + "/data/weka/nominal/sonar.arff"));
				inst.setClassIndex(inst.numAttributes() - 1);
				ho.setDataSet(new WekaInstancesDataSet(inst, 1));
				ho.setModel(new RandomForestModel());
				ho.runSequentially();
				Predictions p = ho.getResult();
				for (PredictionUtil.ClassificationMeasure m : PredictionUtil.ClassificationMeasure
						.values())
				{
					double v = valP.containsKey(m) ? valP.get(m) : 0;
					double v2 = PredictionUtil.getClassificationMeasure(p, m, 1.0);
					valP.put(m, (v * run + v2) / (run + 1));
				}

				Instances instX = new Instances(inst);
				instX.randomize(new Random(run));
				int trainSize = (int) Math.round(instX.numInstances() * 0.66);
				int testSize = instX.numInstances() - trainSize;
				Instances train = new Instances(instX, 0, trainSize);
				Instances test = new Instances(instX, trainSize, testSize);
				RandomForest rf = new RandomForest();
				rf.buildClassifier(train);
				Evaluation eval = new Evaluation(inst);
				eval.evaluateModel(rf, test);
				Assert.assertEquals(p.actual.length, (int) eval.numInstances());
				for (PredictionUtil.ClassificationMeasure m : PredictionUtil.ClassificationMeasure
						.values())
				{
					double v = valE.containsKey(m) ? valE.get(m) : 0;
					double v2 = PredictionUtil.getClassificationMeasureInWeka(eval, m, 1.0);
					valE.put(m, (v * run + v2) / (run + 1));
				}

				maxDiff = 0;
				for (PredictionUtil.ClassificationMeasure m : PredictionUtil.ClassificationMeasure
						.values())
					maxDiff = Math.max(Math.abs(valP.get(m) - valE.get(m)), maxDiff);
				run++;
				System.out.println(run + " " + maxDiff);
			}
			while (maxDiff > 0.05);
			//while (maxDiff > 0.005);

			System.out.println("Holdout stats about equal");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

}
