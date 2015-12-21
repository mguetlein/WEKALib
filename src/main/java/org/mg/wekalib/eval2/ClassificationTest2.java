package org.mg.wekalib.eval2;

import java.awt.Dimension;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.mg.javalib.datamining.ResultSet;
import org.mg.javalib.datamining.ResultSetBoxPlot;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekautil.Predictions;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Instances;

public class ClassificationTest2
{
	public static void main(String[] args) throws Exception
	{
		String datasets[] = new String[] { "anneal", "anneal.ORIG", "audiology", "autos", "breast-cancer", "breast-w",
				"colic", "colic.ORIG", "credit-a", "credit-g", "diabetes", "glass", "heart-c", "heart-h",
				"heart-statlog", "hypothyroid", "ionosphere", "lymph", "primary-tumor", "segment", "sonar", "soybean",
				"vehicle", "vote", "vowel", "zoo" };
		//too-big: "letter", "kr-vs-kp", "splice", waveform-5000, "sick"
		//too-easy: "mushroom"
		//too-unstable: "hepatitis", "labor"
		//not-working: "iris" "balance-scale
		ArrayUtil.scramble(datasets);

		Thread th = new Thread(new Runnable()
		{
			@Override
			public void run()
			{
				while (true)
				{
					String measure = "AUPRC";
					ResultSetBoxPlot bp = new ResultSetBoxPlot(res, "", "Performance", "Algorithm", "Dataset", measure);
					bp.setHideMean(true);
					SwingUtil.showInFrame(bp.getChart(), measure, false, new Dimension(1200, 800));
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
			for (int i = 0; i < datasets.length; i++) //datasets.length
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
			return c.getClass().getSimpleName() + "-" + getName(((SingleClassifierEnhancer) c).getClassifier());
		}
		else
			return c.getClass().getSimpleName();
	}

	public static boolean run(String data, int seed) throws Exception
	{
		Instances inst = new Instances(new FileReader(System.getProperty("user.home") + "/data/weka/nominal/" + data
				+ ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		inst.randomize(new Random(2));
		WekaInstancesDataSet ds = new WekaInstancesDataSet(inst);
		System.out.println(data + " #inst:" + inst.numInstances() + " #feat:" + inst.numAttributes());

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
			List<Model> classifiers = new ArrayList<>();
			for (int numT : new int[] { 10, 100, 200 }) //10, 20, 40, 80, 
			{
				RandomForestModel rf = new RandomForestModel();
				rf.setNumTrees(numT);
				classifiers.add(rf);
			}

			//			Double cs[] = new Double[] { 1.0, 10.0, 100.0 };
			//			for (Double g : new Double[] { 0.001, 0.01, 0.1 })
			//			{
			//				for (Double c : cs)
			//				{
			//					if (c == 1.0 && g == 0.001) // does not work well
			//						continue;
			//					SupportVectorMachineModel svm = new SupportVectorMachineModel();
			//					svm.setC(c);
			//					svm.setKernel(new RBFKernel());
			//					svm.setGamma(g);
			//					classifiers.add(svm);
			//				}
			//			}
			//			for (Double e : new Double[] { 1.0 }) // exponent optimizing not needed , 2.0, 3.0
			//			{
			//				for (Double c : cs)
			//				{
			//					SupportVectorMachineModel svm = new SupportVectorMachineModel();
			//					svm.setC(c);
			//					svm.setKernel(new PolyKernel());
			//					svm.setExp(e);
			//					classifiers.add(svm);
			//				}
			//			}

			for (int i = 0; i < classifiers.size(); i++)
			{
				String name = classifiers.get(i).getName();
				System.out.println(name + " " + seed);

				Instances instX = new Instances(inst);
				instX.randomize(new Random(seed));

				CV cv = new CV();
				cv.setDataSet(ds);
				cv.setModel((Model) classifiers.get(i).cloneJob());
				cv.setNumFolds(10);
				cv.setRandomSeed(seed);
				while (!cv.isDone())
					cv.nextJob().run();
				Predictions p = cv.getResult();

				for (Predictions pf : PredictionUtil.perFold(p))
				{
					//					System.out.println(PredictionUtil.summaryClassification(pf));
					//					System.out.println(PredictionUtil.AUC(pf));
					//					System.exit(1);

					int idx = res.addResult();
					res.setResultValue(idx, "Algorithm", name);
					res.setResultValue(idx, "Dataset", data + " " + inst.numAttributes());
					res.setResultValue(idx, "Fold", pf.fold[0]);
					res.setResultValue(idx, "AUC", PredictionUtil.AUC(pf));
					res.setResultValue(idx, "AUPRC", PredictionUtil.AUPRC(pf));

				}
				//				}
			}
			return true;
		}
	}
}
