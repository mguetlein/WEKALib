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

public class CVTest
{
	@Test
	public void runCVandCompareWithWekaCV()
	{
		try
		{
			HashMap<PredictionUtil.ClassificationMeasure, Double> valP = new HashMap<>();
			HashMap<PredictionUtil.ClassificationMeasure, Double> valE = new HashMap<>();
			int run = 0;
			double maxDiff = 0;

			// run repeated CVs on the same data until the max difference
			// between CV() and Weka-CV
			// in mean statistics for each value is below 0.005
			do
			{
				CV cv = new CV();
				cv.setRandomSeed(run);
				cv.setStratified(true);
				Instances inst = new Instances(new FileReader(
						System.getProperty("user.home") + "/data/weka/nominal/sonar.arff"));
				inst.setClassIndex(inst.numAttributes() - 1);
				cv.setDataSet(new WekaInstancesDataSet(inst, 1));
				cv.setModel(new RandomForestModel());
				cv.runSequentially();
				Predictions p = cv.getResult();
				for (PredictionUtil.ClassificationMeasure m : PredictionUtil.ClassificationMeasure
						.values())
				{
					double v = valP.containsKey(m) ? valP.get(m) : 0;
					double v2 = PredictionUtil.getClassificationMeasure(p, m, 1.0);
					valP.put(m, (v * run + v2) / (run + 1));
				}

				Evaluation eval = new Evaluation(inst);
				RandomForest rf = new RandomForest();
				eval.crossValidateModel(rf, inst, 10, new Random(run), new Object[0]);
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
			while (maxDiff > 0.005);

			System.out.println("CVs stats about equal");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

}
