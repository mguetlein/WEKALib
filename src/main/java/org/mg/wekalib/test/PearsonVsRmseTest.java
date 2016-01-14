package org.mg.wekalib.test;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.mg.javalib.util.StringUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.Predictions;
import org.mg.wekalib.evaluation.PredictionsPlot;

public class PearsonVsRmseTest
{
	static String baseDir = "/home/martin/documents/dream/pearson-vs-rmse/";

	public static void outlier()
	{
		NormalDistribution d = new NormalDistribution(0.0, 1.0);
		NormalDistribution dev = new NormalDistribution(0.0, 1.0);
		Predictions pred = new Predictions();
		for (int i = 0; i < 99; i++)
		{
			double v = d.sample();
			double error = dev.sample();
			double a = v + error;
			double p = v - error;
			PredictionUtil.add(pred, a, p, 1.0, 1, 1);
		}
		Predictions pred2 = PredictionUtil.clone(pred);
		PredictionUtil.add(pred, 10, 18, 0.0, 1, 1);

		PredictionsPlot plot = new PredictionsPlot(new Predictions[] { pred2, pred },
				new String[] { "without-outlier", "with-outlier" });
		plot.setTitle("Outlier");
		plot.show(false);
		plot.plotToPngFile(baseDir + "2_outlier.png");
	}

	public static void deviation(boolean errorEqualsDeviation)
	{
		String file = baseDir + (errorEqualsDeviation ? "1b_errorEqualsDeviation.png"
				: "1a_sameDeviationDifferentError.png");

		String name = "";
		Predictions pred = new Predictions();
		{
			double sd = errorEqualsDeviation ? (2 / 3.0) : 1.0;
			double rmse = 2 / 3.0;
			name = "standard-deviation:" + StringUtil.formatDouble(sd);
			NormalDistribution d = new NormalDistribution(0.0, sd);
			NormalDistribution dev = new NormalDistribution(0.0, rmse);
			for (int i = 0; i < 3000; i++)
			{
				double v = d.sample();
				double error = dev.sample();
				double a = v + error * 0.5;
				double p = v - error * 0.5;
				PredictionUtil.add(pred, a, p, 1, 1, 1);
			}
			//				System.out.println(DoubleArraySummary.create(pred.actual).toStringSummary());
			//				System.out.println(DoubleArraySummary.create(pred.predicted).toStringSummary());
		}

		String name2 = "";
		Predictions pred2 = new Predictions();
		{
			double sd = 1.0;
			double rmse = 1.0;
			name2 = "standard-deviation:" + StringUtil.formatDouble(sd);
			NormalDistribution d = new NormalDistribution(0.0, sd);
			NormalDistribution dev = new NormalDistribution(0.0, rmse);
			for (int i = 0; i < 3000; i++)
			{
				double v = d.sample();
				double error = dev.sample();
				double a = v + error * 0.5;
				double p = v - error * 0.5;
				PredictionUtil.add(pred2, a, p, 1, 1, 1);
			}
		}
		PredictionsPlot plot = new PredictionsPlot(new Predictions[] { pred, pred2 },
				new String[] { name, name2 });
		plot.setTitle(errorEqualsDeviation ? "Error = Standard deviation"
				: "Same deviation, different error");
		plot.show(false);
		plot.plotToPngFile(file);
	}

	public static void unbalanced()
	{
		NormalDistribution d = new NormalDistribution(0.0, 1.0);
		NormalDistribution dev1 = new NormalDistribution(0.9, 0.5);
		NormalDistribution dev2 = new NormalDistribution(-0.9, 0.5);
		Predictions pred = new Predictions();
		for (int i = 0; i < 1000; i++)
		{
			double v = d.sample();
			double error;
			if (v < -1)
				error = dev1.sample();
			else if (v < 1)
				error = dev2.sample();
			else
				error = dev1.sample();
			double a = v + error * 0.5;
			double p = v - error * 0.5;
			PredictionUtil.add(pred, a, p, 1.0, 1, 1);
		}

		Predictions pred2 = new Predictions();
		NormalDistribution dev3 = new NormalDistribution(0.0, 1.0);
		for (int i = 0; i < 1000; i++)
		{
			double v = d.sample();
			double error = dev3.sample();
			double a = v + error * 0.5;
			double p = v - error * 0.5;
			PredictionUtil.add(pred2, a, p, 1.0, 1, 1);
		}

		PredictionsPlot plot = new PredictionsPlot(new Predictions[] { pred, pred2 },
				new String[] { "unbalanced", "straight" });
		plot.setTitle("Unbalanced");
		plot.show(false);
		plot.plotToPngFile(baseDir + "3_unbalanced.png");
	}

	public static void main(String[] args)
	{
		outlier();
		deviation(true);
		deviation(false);
		unbalanced();

		//		// wurst
		//		{
		//			Predictions pred = new Predictions();
		//			PredictionUtil.add(pred, 0, 1, 0, 1, 1);
		//			PredictionUtil.add(pred, 1, 2, 0, 1, 1);
		//			PredictionUtil.add(pred, 2, 3, 0, 1, 1);
		//			PredictionUtil.add(pred, 3, 4, 0, 1, 1);
		//			PredictionUtil.add(pred, 4, 5, 0, 1, 1);
		//			PredictionUtil.add(pred, 5, 6, 0, 1, 1);
		//			PredictionUtil.add(pred, 6, 7, 0, 1, 1);
		//			PredictionUtil.add(pred, 7, 8, 0, 1, 1);
		//			PredictionUtil.add(pred, 8, 9, 0, 1, 1);
		//
		//			Predictions pred2 = new Predictions();
		//			PredictionUtil.add(pred2, 2, 1, 1, 2, 1);
		//			PredictionUtil.add(pred2, 2, 2, 1, 2, 1);
		//			PredictionUtil.add(pred2, 2, 3, 1, 2, 1);
		//			PredictionUtil.add(pred2, 3, 4, 1, 2, 1);
		//			PredictionUtil.add(pred2, 4, 5, 1, 2, 1);
		//			PredictionUtil.add(pred2, 5, 6, 1, 2, 1);
		//			PredictionUtil.add(pred2, 6, 7, 1, 2, 1);
		//			PredictionUtil.add(pred2, 7, 7, 1, 2, 1);
		//			PredictionUtil.add(pred2, 8, 7, 1, 2, 1);
		//
		//			PredictionUtil.plot("Wurst", "", new Predictions[] { pred, pred2 }, new String[] { "straight", "curved" });
		//		}

		//		{
		//			Random r = new Random();
		//			List<Predictions> p = new ArrayList<>();
		//			List<String> n = new ArrayList<>();
		//			for (int i = 0; i < 2; i++)
		//			{
		//				NormalDistribution norm = new NormalDistribution(r.nextDouble(), r.nextDouble());
		//				Predictions pred = new Predictions();
		//				for (int j = 0; j < 50; j++)
		//					PredictionUtil.add(pred, norm.sample(), norm.sample(), 1, 1, 1);
		//				p.add(pred);
		//				n.add(i + "");
		//			}
		//			PredictionUtil.plot("", "", ArrayUtil.toArray(p), ArrayUtil.toArray(n));
		//		}

		//		{
		//			NormalDistribution d = new NormalDistribution(1.0, 0.5);
		//			NormalDistribution devA = new NormalDistribution(-0.3, 0.3);
		//			NormalDistribution devB = new NormalDistribution(0.0, 0.3);
		//			NormalDistribution devC = new NormalDistribution(+0.3, 0.3);
		//			Predictions pred = new Predictions();
		//			Random r = new Random();
		//			for (int i = 0; i < 1000; i++)
		//			{
		//				double a = d.sample();
		//				double p;
		//				if (a < 0.5)
		//					p = a + devA.sample();
		//				else if (a > 1.5)
		//					p = a + devA.sample();
		//				else
		//					p = a + devC.sample();
		//				double c = r.nextDouble();
		//				PredictionUtil.add(pred, a, p, c, 1, 1);
		//			}
		//			PredictionUtil.plot(pred);
		//		}
		SwingUtil.waitWhileWindowsVisible();
		System.exit(0);
	}
}
