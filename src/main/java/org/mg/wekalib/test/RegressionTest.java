package org.mg.wekalib.test;

import java.awt.Dimension;
import java.io.FileReader;
import java.util.Random;

import org.mg.javalib.datamining.ResultSet;
import org.mg.javalib.datamining.ResultSetBoxPlot;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.classifier.RegressionByDiscretizationW;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.Predictions;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.estimators.UnivariateNormalEstimator;

public class RegressionTest
{
	public static void main(String[] args) throws Exception
	{
		String datasets[] = new String[] { "anneal", "anneal.ORIG", "audiology", "autos",
				"balance-scale", "breast-cancer", "breast-w", "colic", "colic.ORIG", "credit-a",
				"credit-g", "diabetes", "glass", "heart-c", "heart-h", "heart-statlog", "hepatitis",
				"hypothyroid.arff", "ionosphere", "iris", "kr-vs-kp", "labor", "letter", "lymph",
				"mushroom", "primary-tumor", "segment", "sick", "sonar", "soybean", "splice",
				"vehicle", "vote", "vowel", "waveform-5000", "zoo" };
		//too-big: "quake"
		//unstable/not-modable: "fruitfly","breastTumor", "bolts"
		ArrayUtil.scramble(datasets);

		Thread th = new Thread(new Runnable()
		{

			@Override
			public void run()
			{
				while (true)
				{
					ResultSetBoxPlot bp = new ResultSetBoxPlot(res, "", "Performance", "Algorithm",
							"Dataset", "Pearson");
					bp.setHideMean(true);

					SwingUtil.showInFrame(bp.getChart(), "Pearson", false,
							new Dimension(1200, 800));
					SwingUtil.waitWhileWindowsVisible();
				}

				//				for (Window w : Window.getWindows())
				//					w.dispose();
				//				SwingUtil.waitForAWTEventThread();

				//				ResultSetBoxPlot bp = new ResultSetBoxPlot(res, "", "Performance", "Algorithm", "Dataset", "Pearson");
				//				bp.setHideMean(true);
				//				SwingUtil.showInFrame(bp.getChart(), "Pearson", false);

				//					bp = new ResultSetBoxPlot(res, "", "Performance", "Algorithm", "Dataset", "RMSE");
				//					bp.setHideMean(true);
				//					SwingUtil.showInFrame(bp.getChart(), "RMSE", false);

				//				ScreenUtil.centerWindowsOnScreen();
			}
		});
		th.start();

		for (int seed = 0; seed < 10; seed++)
		{
			for (int i = 0; i < 15; i++) //datasets.length
			{
				run(datasets[i], seed);
			}
		}
		//		SwingUtil.waitWhileWindowsVisible();
		//		System.exit(1);
	}

	static ResultSet res = new ResultSet();

	public static String getName(Classifier c)
	{
		if (c instanceof SingleClassifierEnhancer)
		{
			return c.getClass().getSimpleName() + "-"
					+ getName(((SingleClassifierEnhancer) c).getClassifier());
		}
		else
			return c.getClass().getSimpleName();
	}

	public static boolean run(String data, int seed) throws Exception
	{
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/numeric/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		inst.randomize(new Random(2));

		System.out
				.println(data + " #inst:" + inst.numInstances() + " #feat:" + inst.numAttributes());

		if (inst.numInstances() < 30)
		{
			System.out.println("too small");
			return false;
		}
		else
		{
			//			Classifier class1 = new M5Rules();

			RegressionByDiscretizationW class1 = new RegressionByDiscretizationW();
			class1.setClassifier(new RandomForest());
			class1.setNumBins(15);
			class1.setEstimator(new UnivariateNormalEstimator());

			Bagging bag1 = new Bagging();
			bag1.setClassifier(class1);

			LWL lwl1 = new LWL();
			lwl1.setClassifier(bag1);
			lwl1.setKNN(-1);//(int) (inst.numInstances() * 0.5));

			//			RegressionByDiscretization regr = new RegressionByDiscretization();
			//			regr.setClassifier(new RandomForest());
			//			regr.setNumBins(15);
			//			regr.setEstimator(new UnivariateNormalEstimator());
			//
			//			Bagging bag2 = new Bagging();
			//			bag2.setClassifier(regr);
			//
			//			LWL lwl2 = new LWL();
			//			lwl2.setClassifier(bag2);

			for (Classifier classifier : new Classifier[] { class1, lwl1 })//, class2, bag2, lwl2 })
			{
				String name = getName(classifier);
				System.out.println(name + " " + seed);

				//				for (int i = 0; i < 3; i++)
				//				{
				//					System.out.println("rep " + i);

				Instances instX = new Instances(inst);
				instX.randomize(new Random(seed));

				int trainSize = (int) Math.round(instX.numInstances() * 0.95);
				int testSize = instX.numInstances() - trainSize;
				Instances train = new Instances(instX, 0, trainSize);
				Instances test = new Instances(instX, trainSize, testSize);
				//				System.out.println(test);
				classifier.buildClassifier(train);

				CVPredictionsEvaluation eval = new CVPredictionsEvaluation(train);
				eval.evaluateModel(classifier, test);

				//				eval.crossValidateModel(classifier, inst, 5, new Random(1), new Object[0]);

				Predictions p = eval.getCvPredictions();

				for (Predictions pf : PredictionUtil.perFold(p))
				{
					int idx = res.addResult();
					res.setResultValue(idx, "Algorithm", name);
					res.setResultValue(idx, "Dataset", data);
					res.setResultValue(idx, "Fold", pf.fold[0]);
					res.setResultValue(idx, "RMSE", PredictionUtil.rmse(pf));
					res.setResultValue(idx, "Pearson", PredictionUtil.pearson(pf));
				}
				//				}
			}
			return true;
		}
	}
}
