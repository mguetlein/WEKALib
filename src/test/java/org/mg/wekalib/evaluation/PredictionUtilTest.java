package org.mg.wekalib.evaluation;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.mg.wekalib.eval2.CV;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.model.RandomForestModel;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class PredictionUtilTest
{
	public static final double DELTA_EQ = 1e-15;

	@Test
	public void compareWithWekaEval()
	{
		try
		{
			String data = "breast-cancer";
			long seed = 2L;
			Instances inst = new Instances(new FileReader(
					System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
			inst.randomize(new Random(seed));
			inst.setClassIndex(inst.numAttributes() - 1);
			System.out.println(inst.classAttribute().toString());

			Assert.assertTrue(inst.numClasses() == 2);

			int trainSize = (int) Math.round(inst.numInstances() * 0.90);
			int testSize = inst.numInstances() - trainSize;
			Instances train = new Instances(inst, 0, trainSize);
			Instances test = new Instances(inst, trainSize, testSize);

			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(train);

			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(nb, test);

			CVPredictionsEvaluation eval2 = new CVPredictionsEvaluation(train);
			eval2.evaluateModel(nb, test);
			Predictions p = eval2.getCvPredictions();

			Assert.assertEquals(p.actual.length, (int) eval.numInstances());
			Assert.assertEquals(eval.areaUnderROC(0), eval.areaUnderROC(1), DELTA_EQ);
			Assert.assertNotEquals(eval.areaUnderPRC(0), eval.areaUnderPRC(1), DELTA_EQ);

			ArrayList<Prediction> ps = eval.predictions();
			for (int i = 0; i < ps.size(); i++)
			{
				NominalPrediction pi = (NominalPrediction) ps.get(i);
				Assert.assertEquals(pi.actual(), p.actual[i], DELTA_EQ);
				Assert.assertEquals(pi.predicted(), p.predicted[i], DELTA_EQ);
				Assert.assertEquals(pi.distribution()[0], 1 - pi.distribution()[1], DELTA_EQ);
				if (pi.predicted() == 0.0)
					Assert.assertEquals(pi.distribution()[0], p.confidence[i] * 0.5 + 0.5,
							DELTA_EQ);
				else if (pi.predicted() == 1.0)
					Assert.assertEquals(pi.distribution()[1], p.confidence[i] * 0.5 + 0.5,
							DELTA_EQ);
			}

			equal(p, eval, DELTA_EQ);

			System.out.println("test passed: compareWithWekaEval");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}

	private void equal(Predictions p, Evaluation eval, double delta)
	{
		for (double positveClass : new Double[] { 0.0, 1.0 })
		{
			//			System.out.println("\npostivive class: " + positveClass);

			for (PredictionUtil.ClassificationMeasure m : PredictionUtil.ClassificationMeasure
					.values())
			{
				Double val = PredictionUtil.getClassificationMeasure(p, m, positveClass);
				Double val2 = PredictionUtil.getClassificationMeasureInWeka(eval, m, positveClass);
				//System.out.println(m + " " + val + " +" + val2);
				Assert.assertEquals(val, val2, delta);
			}
		}
	}

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

	public static void main(String[] args)
	{
		new PredictionUtilTest().runCVandCompareWithWekaCV();
	}
}
