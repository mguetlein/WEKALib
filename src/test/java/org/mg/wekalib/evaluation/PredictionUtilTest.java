package org.mg.wekalib.evaluation;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;
import org.mg.wekalib.classifier.AbstainingClassifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
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

			Classifier c;

			c = new AbstainingClassifier();
			((AbstainingClassifier) c).setClassifier(new NaiveBayes());

			c.buildClassifier(train);

			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(c, test);

			PredictionsEvaluation eval2 = new PredictionsEvaluation(train);
			eval2.evaluateModel(c, test);
			Predictions p = eval2.getCvPredictions();

			Assert.assertEquals(p.actual.length, (int) eval.numInstances());
			//			Assert.assertEquals(eval.areaUnderROC(0), eval.areaUnderROC(1), DELTA_EQ);
			//			Assert.assertNotEquals(eval.areaUnderPRC(0), eval.areaUnderPRC(1), DELTA_EQ);

			ArrayList<Prediction> ps = eval.predictions();
			for (int i = 0; i < ps.size(); i++)
			{
				NominalPrediction pi = (NominalPrediction) ps.get(i);
				Assert.assertEquals(pi.actual(), p.actual[i], DELTA_EQ);
				Assert.assertEquals(pi.predicted(), p.predicted[i], DELTA_EQ);
				//Assert.assertEquals(pi.distribution()[0], 1 - pi.distribution()[1], DELTA_EQ);
				if (pi.predicted() == 0.0)
					Assert.assertEquals(pi.distribution()[0], p.confidence[i] * 0.5 + 0.5,
							DELTA_EQ);
				else if (pi.predicted() == 1.0)
					Assert.assertEquals(pi.distribution()[1], p.confidence[i] * 0.5 + 0.5,
							DELTA_EQ);
			}

			//			SwingUtil.showInFrame(PredictionUtilPlots.getROCPlot(p, 1));

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
				try
				{
					Double val2 = PredictionUtil.getClassificationMeasureInWeka(eval, m,
							positveClass);
					Double val = PredictionUtil.getClassificationMeasure(p, m, positveClass);
					System.out.println(m + " own: " + val + " weka: " + val2);
					Assert.assertEquals(val, val2, delta);
				}
				catch (IllegalArgumentException e)
				{
					System.err.println(e.getMessage());
				}
			}
		}
	}

	public static void main(String[] args)
	{
		new PredictionUtilTest().compareWithWekaEval();
	}
}
