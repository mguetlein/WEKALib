package org.mg.wekalib.test;

import java.awt.Dimension;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mg.javalib.datamining.ResultSet;
import org.mg.javalib.datamining.ResultSetBoxPlot;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.Predictions;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class ClassificationTest
{
	public static void main(String[] args) throws Exception
	{
		String datasets[] = new String[] { "anneal", "anneal.ORIG", "audiology", "autos",
				"breast-cancer", "breast-w", "colic", "colic.ORIG", "credit-a", "credit-g",
				"diabetes", "glass", "heart-c", "heart-h", "heart-statlog", "hypothyroid",
				"ionosphere", "labor", "lymph", "primary-tumor", "segment", "sonar", "soybean",
				"vehicle", "vote", "vowel", "zoo" };
		//too-big: "letter", "kr-vs-kp", "splice", waveform-5000, "sick"
		//too-easy: "mushroom"
		//too-unstable: "hepatitis"
		//not-working: "iris" "balance-scale
		ArrayUtil.scramble(datasets);

		Thread th = new Thread(new Runnable()
		{
			@Override
			public void run()
			{
				while (true)
				{
					ResultSetBoxPlot bp = new ResultSetBoxPlot(res, "", "Performance", "Algorithm",
							"Dataset", "AUC");
					bp.setHideMean(true);
					SwingUtil.showInFrame(bp.getChart(), "AUC", false, new Dimension(1200, 800));
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
			for (int i = 0; i < 25; i++) //datasets.length
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
				System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		inst.randomize(new Random(2));
		System.out
				.println(data + " #inst:" + inst.numInstances() + " #feat:" + inst.numAttributes());

		if (inst.numInstances() < 30)
		{
			System.out.println("too small");
			return false;
		}
		else if (inst.classAttribute().numValues() > 3)
		{
			System.out.println("not binary");
			return false;
		}
		else
		{
			List<Classifier> classifiers = new ArrayList<>();
			List<String> names = new ArrayList<>();
			for (boolean rbf : new boolean[] { true, false })
			{
				for (Double c : new Double[] { 1.0, 10.0, 100.0 })
				{
					for (Double g : new Double[] { 0.001, 0.01, 0.1 })
					{
						for (Double e : new Double[] { 1.0, 2.0, 3.0 })
						{
							SMO smo = new SMO();
							smo.setC(c);
							String name = "SMO ";
							name += rbf ? " rbf" : " poly";
							name += " c" + c;
							if (rbf && c == 0.01)
								continue;
							//							if (rbf && c == 0.1 && g == 0.01)
							//								continue;
							if (!rbf && g != 0.01)
								continue;
							if (rbf && e != 1.0)
								continue;
							smo.setKernel(rbf ? new RBFKernel() : new PolyKernel());
							if (rbf)
							{
								((RBFKernel) smo.getKernel()).setGamma(g);
								name += " g" + g;
							}
							else
							{
								((PolyKernel) smo.getKernel()).setExponent(e);
								name += " e" + e;
							}
							classifiers.add(smo);
							names.add(name);
						}
					}
				}
			}
			//			classifiers.add(new RandomForest());
			//			names.add("Random Forest 100");
			//
			//			RandomForest rf = new RandomForest();
			//			rf.setNumTrees(1000);
			//			classifiers.add(rf);
			//			names.add("Random Forest 1000");
			//
			//			classifiers.add(new NaiveBayes());
			//			names.add("Naive Bayes");

			for (int i = 0; i < classifiers.size(); i++)
			{
				Classifier classifier = classifiers.get(i);
				String name = names.get(i);

				//				String name = getName(classifier);
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
					//					System.out.println(PredictionUtil.summaryClassification(pf));
					//					System.out.println(PredictionUtil.AUC(pf));
					//					System.exit(1);

					int idx = res.addResult();
					res.setResultValue(idx, "Algorithm", name);
					res.setResultValue(idx, "Dataset", data);
					res.setResultValue(idx, "Fold", pf.fold[0]);
					res.setResultValue(idx, "AUC", PredictionUtil.AUC(pf));

				}
				//				}
			}
			return true;
		}
	}
}
