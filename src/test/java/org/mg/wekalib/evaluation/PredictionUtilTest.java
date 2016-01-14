package org.mg.wekalib.evaluation;

import java.io.FileReader;
import java.util.ArrayList;
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
				//				System.out.println(m + " " + val);
				switch (m)
				{
					case accuracy:
						Assert.assertEquals(val, eval.pctCorrect() / 100.0, delta);
						break;
					case AUC:
						Assert.assertEquals(val, eval.areaUnderROC((int) positveClass), delta);
						break;
					case AUPRC:
						Assert.assertEquals(val, eval.areaUnderPRC((int) positveClass), delta);
						break;
					case sensitivity:
						Assert.assertEquals(val, eval.truePositiveRate((int) positveClass), delta);
						break;
					case specificity:
						Assert.assertEquals(val, eval.trueNegativeRate((int) positveClass), delta);
						break;
					default:
						throw new IllegalStateException("add test for " + m);
				}
			}
		}
	}

	@Test
	public void runCVandCompareWithWekaCV()
	{
		try
		{
			long seed = -483616214247688869L; //new Random().nextLong();
			//System.err.println(seed);

			CV cv = new CV();
			cv.setRandomSeed(seed);
			Instances inst = new Instances(new FileReader(
					System.getProperty("user.home") + "/data/weka/nominal/sonar.arff"));
			inst.setClassIndex(inst.numAttributes() - 1);
			cv.setDataSet(new WekaInstancesDataSet(inst, 1));
			cv.setModel(new RandomForestModel());
			cv.runSequentially();
			Predictions p = cv.getResult();

			Evaluation eval = new Evaluation(inst);
			RandomForest rf = new RandomForest();
			eval.crossValidateModel(rf, inst, 10, new Random(seed), new Object[0]);

			Assert.assertEquals(p.actual.length, (int) eval.numInstances());
			Assert.assertEquals(eval.areaUnderROC(0), eval.areaUnderROC(1), DELTA_EQ);
			Assert.assertNotEquals(eval.areaUnderPRC(0), eval.areaUnderPRC(1), DELTA_EQ);

			equal(p, eval, 0.04);

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
