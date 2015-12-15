package org.mg.wekalib.test;

import java.awt.Dimension;
import java.awt.Font;
import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.annotations.XYAnnotation;
import org.jfree.chart.annotations.XYLineAnnotation;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.DefaultXYDataset;
import org.mg.javalib.freechart.FreeChartUtil;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekalib.classifier.MyBagging;
import org.mg.wekalib.evaluation.CVPredictionsEvaluation;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.PredictionsPlot;
import org.mg.wekautil.Predictions;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class CondDensTest
{
	public static void main(String[] args) throws Exception
	{
		for (String s : new String[] { "auto93.arff", "autoHorse.arff", "autoMpg.arff", "autoPrice.arff",
				"baskball.arff", "bodyfat.arff", "bolts.arff", "breastTumor.arff", "cholesterol.arff",
				"cleveland.arff", "cloud.arff", "cpu.arff", "detroit.arff", "echoMonths.arff", "elusage.arff",
				"fishcatch.arff", "fruitfly.arff", "gascons.arff", "housing.arff", "hungarian.arff", "longley.arff",
				"lowbwt.arff", "mbagrade.arff", "meta.arff", "pbc.arff", "pharynx.arff", "pollution.arff",
				"pwLinear.arff", "quake.arff", "schlvote.arff", "sensory.arff", "servo.arff", "sleep.arff",
				"strike.arff", "veteran.arff", "vineyard.arff" })
		{
			testDensitiy(System.getProperty("user.home") + "/data/weka/numeric/" + s);

		}
		printMedianDiffs();
		SwingUtil.waitWhileWindowsVisible();
	}

	static DescriptiveStatistics diffRMSE = new DescriptiveStatistics();
	static DescriptiveStatistics diffCorr = new DescriptiveStatistics();

	public static void printMedianDiffs()
	{
		System.out.println(diffRMSE.getValues().length);
		System.out.println("Median RMSE diff: " + diffRMSE.getPercentile(50));
		System.out.println("Median Corr diff: " + diffCorr.getPercentile(50));
	}

	public static void plot(String filePrefix, String title, String subtitles, List<Double> actualAll,
			List<Double> predictedAll, List<Double> actualReduced, List<Double> predictedReduced)
	{
		DefaultXYDataset d = new DefaultXYDataset();

		if (predictedReduced != null)
			d.addSeries(
					"reduced",
					new double[][] { ArrayUtil.toPrimitiveDoubleArray(predictedReduced),
							ArrayUtil.toPrimitiveDoubleArray(actualReduced) });

		d.addSeries(
				"all",
				new double[][] { ArrayUtil.toPrimitiveDoubleArray(predictedAll),
						ArrayUtil.toPrimitiveDoubleArray(actualAll) });

		//		PearsonsCorrelation p = new PearsonsCorrelation();
		//		System.out.println("all "
		//				+ p.correlation(ArrayUtil.toPrimitiveDoubleArray(predictedAll),
		//						ArrayUtil.toPrimitiveDoubleArray(actualAll)));
		//		System.out.println("reduced "
		//				+ p.correlation(ArrayUtil.toPrimitiveDoubleArray(predictedReduced),
		//						ArrayUtil.toPrimitiveDoubleArray(actualReduced)));

		JFreeChart f = ChartFactory.createScatterPlot("title", "predicted", "actual", d);
		XYPlot plot = f.getXYPlot();
		//					for (String k : actual.keySet())
		//					{
		//						Shape cross = ShapeUtilities.createDiagonalCross(3, 1);
		//		XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();
		//		renderer.setBaseShapesFilled(false);
		//						renderer.setSeriesShape(0, cross);
		//					}
		ValueAxis yAxis = plot.getRangeAxis();
		ValueAxis xAxis = plot.getDomainAxis();
		double min = Math.min(xAxis.getRange().getLowerBound(), yAxis.getRange().getLowerBound());
		double max = Math.max(xAxis.getRange().getUpperBound(), yAxis.getRange().getUpperBound());
		yAxis.setRange(min, max);
		xAxis.setRange(min, max);
		XYAnnotation diagonal = new XYLineAnnotation(xAxis.getRange().getLowerBound(),
				yAxis.getRange().getLowerBound(), xAxis.getRange().getUpperBound(), yAxis.getRange().getUpperBound());
		plot.addAnnotation(diagonal);
		ChartPanel p = new ChartPanel(f);
		p.getChart().setTitle(title);

		p.getChart().setSubtitles(ListUtil.createList(new TextTitle(subtitles)));

		final StandardChartTheme chartTheme = (StandardChartTheme) org.jfree.chart.StandardChartTheme
				.createJFreeTheme();

		final Font oldExtraLargeFont = chartTheme.getExtraLargeFont();
		final Font oldLargeFont = chartTheme.getLargeFont();
		final Font oldRegularFont = chartTheme.getRegularFont();
		final Font oldSmallFont = chartTheme.getSmallFont();

		final Font extraLargeFont = new Font("Monospaced", oldExtraLargeFont.getStyle(), oldExtraLargeFont.getSize());
		final Font largeFont = new Font("Monospaced", oldLargeFont.getStyle(), oldLargeFont.getSize());
		final Font regularFont = new Font("Monospaced", oldRegularFont.getStyle(), oldRegularFont.getSize());
		final Font smallFont = new Font("Monospaced", oldSmallFont.getStyle(), oldSmallFont.getSize());

		chartTheme.setExtraLargeFont(extraLargeFont);
		chartTheme.setLargeFont(largeFont);
		chartTheme.setRegularFont(regularFont);
		chartTheme.setSmallFont(smallFont);

		chartTheme.apply(p.getChart());

		FreeChartUtil.toPNGFile("/tmp/density/" + filePrefix + "_" + title + ".png", p, new Dimension(700, 900));

		SwingUtil.showInFrame(p, "Plot", false, new Dimension(600, 800));
	}

	public static void testDensitiy(String data) throws Exception
	{
		String subtitle = "";
		//Instances inst = new Instances(new FileReader("/home/martin/data/workspace/external/weka-3-7-12/data/cpu.arff"));
		Instances inst = new Instances(new FileReader(data));
		inst.randomize(new Random(2));

		subtitle += "all        #i:" + inst.numInstances() + "\n";

		if (inst.numInstances() < 30)
		{
			System.out.println("too small");
		}
		else
		{
			//			Normalize norm = new Normalize();
			//			norm.setInputFormat(inst);
			//			norm.setTranslation(0);
			//			//norm.setTranslation(-0);
			//			//norm.setIgnoreClass(false);
			//			inst = Filter.useFilter(inst, norm);

			inst.setClassIndex(inst.numAttributes() - 1);

			double min = Double.MAX_VALUE;
			for (Instance i : inst)
				min = Math.min(min, i.classValue());
			if (min <= 0)
				return;
			//			if (min < 0)
			//				for (Instance i : inst)
			//					i.setClassValue(i.classValue() + Math.abs(min));
			for (Instance i : inst)
				i.setClassValue(Math.log(i.classValue()));

			//			MathExpression exp = new MathExpression();
			//			exp.setIgnoreRange("last");
			//			exp.setExpression("log(A)");
			//			exp.setInputFormat(inst);
			//			inst = Filter.useFilter(inst, exp);

			//			RegressionByDiscretization classifier = new RegressionByDiscretization();
			//			classifier.setClassifier(new RandomForest());
			//			classifier.setEstimator(new UnivariateNormalEstimator());
			// histo: -9.5 1.0
			// kernel: -22.0 0.5 
			// normal: -21.5 2.0

			MyBagging classifier = new MyBagging();
			classifier.setClassifier(new IBk(3));

			//			GaussianProcesses classifier = new GaussianProcesses();

			//			class MyKNN extends IBk implements ConditionalDensityEstimator
			//			{
			//				public MyKNN()
			//				{
			//					super(3);
			//				}
			//
			//				@Override
			//				public double logDensity(Instance instance, double value) throws Exception
			//				{
			//					if (m_Train.numInstances() == 0)
			//					{
			//						throw new IllegalArgumentException();
			//					}
			//					if ((m_WindowSize > 0) && (m_Train.numInstances() > m_WindowSize))
			//					{
			//						m_kNNValid = false;
			//						boolean deletedInstance = false;
			//						while (m_Train.numInstances() > m_WindowSize)
			//						{
			//							m_Train.delete(0);
			//						}
			//						//rebuild datastructure KDTree currently can't delete
			//						if (deletedInstance == true)
			//							m_NNSearch.setInstances(m_Train);
			//					}
			//
			//					// Select k by cross validation
			//					if (!m_kNNValid && (m_CrossValidate) && (m_kNNUpper >= 1))
			//					{
			//						crossValidate();
			//					}
			//
			//					m_NNSearch.addInstanceInfo(instance);
			//
			//					Instances neighbours = m_NNSearch.kNearestNeighbours(instance, m_kNN);
			//
			//					DescriptiveStatistics desc = new DescriptiveStatistics();
			//					for (int i = 0; i < neighbours.size(); i++)
			//						desc.addValue(neighbours.get(i).classValue());
			//					return -desc.getVariance();
			//				}
			//			}
			//			MyKNN classifier = new MyKNN();

			//			MyRegressionSplitEvaluator eval = new MyRegressionSplitEvaluator();
			//			Evaluation ev = new Evaluation(inst)
			//			{
			//
			//			};
			//			ev.crossValidateModel(classifier, inst, 10, new Random(), new Object[0]);

			double r1;
			double c1;
			double r2;
			double c2;
			int dr;
			int dc;

			{
				CVPredictionsEvaluation eval = new CVPredictionsEvaluation(inst);
				eval.crossValidateModel(classifier, inst, 10, new Random(1), new Object[0]);
				Predictions p = eval.getCvPredictions();
				p = PredictionUtil.stripActualNaN(p);
				//				System.out.println(ArrayUtil.toString(p.actual));
				//				System.out.println(ArrayUtil.toString(p.predicted));
				//				System.out.println(ArrayUtil.toString(p.confidence));
				//				System.out.println(ArrayUtil.toString(p.fold));
				//				System.out.println(ArrayUtil.toString(p.origIndex));
				r1 = PredictionUtil.rmse(p);
				c1 = PredictionUtil.pearson(p);

				if (c1 < 0.2)// || c1 > 0.9)
					return;

				//				eval = new MyEvaluation(inst);
				//				eval.crossValidateModel(classifier2, inst, 10, new Random(1), new Object[0]);
				//				Predictions pTop = eval.getCvPredictions();
				//				pTop = PredictionUtil.stripActualNaN(pTop);

				Predictions pTop = PredictionUtil.topConf(p, 0.3);
				r2 = PredictionUtil.rmse(pTop);
				c2 = PredictionUtil.pearson(pTop);

				dr = (int) (((r2 - r1) / r1) * 100);
				dc = (int) (((c2 - c1) / c1) * 100);

				System.out.println();
				System.out.println(new File(data).getName().replace(".arff", "") + " #i:" + inst.size());
				System.out.println("             RMSE pearson");
				System.out.println("test       " + String.format("%.2f", r1) + " " + String.format("%.2f", c1) + "");
				System.out.println("test-top   " + String.format("%.2f", r2) + " " + String.format("%.2f", c2) + "");
				System.out.println("% diff    " + String.format("%4d", dr) + " " + String.format("%4d", dc));

				PredictionsPlot plot = new PredictionsPlot(p);
				plot.setTitle(new File(data).getName().replace(".arff", ""));
				plot.setSubtitle("Instances: " + inst.size() + "\n - ");
				plot.show(false);
			}
			//			{
			//				int trainSize = (int) Math.round(inst.numInstances() * 0.70);
			//				int testSize = inst.numInstances() - trainSize;
			//				Instances train = new Instances(inst, 0, trainSize);
			//				Instances test = new Instances(inst, trainSize, testSize);
			//				//		System.out.println("Train");
			//				subtitle += "train      #i:" + train.numInstances() + "\n";
			//
			//				classifier.buildClassifier(train);
			//				//		System.out.println("Test");
			//				subtitle += "test       #i:" + test.numInstances() + "\n";
			//				Evaluation eval = new Evaluation(test);
			//				eval.evaluateModel(classifier, test);
			//
			//				r1 = eval.rootMeanSquaredError();
			//				c1 = eval.correlationCoefficient();
			//				DescriptiveStatistics stats = new DescriptiveStatistics();
			//				double logDensity[] = new double[test.size()];
			//
			//				if (c1 < 0.1)// || c1 <= 0.9)
			//					return;
			//
			//				List<Double> actAll = new ArrayList<>();
			//				List<Double> predAll = new ArrayList<>();
			//
			//				for (int i = 0; i < test.numInstances(); i++)
			//				{
			//					double d = classifier.classifyInstance(test.get(i));
			//					logDensity[i] = classifier.logDensity(test.get(i), d);
			//					stats.addValue(logDensity[i]);
			//
			//					actAll.add(test.get(i).value(inst.numAttributes() - 1));
			//					predAll.add(d);
			//				}
			//				double perc = 30;
			//				//		System.out.println("Percentile: " + perc);
			//				double thres = stats.getPercentile(100 - perc);
			//				//		System.out.println("Log dens threshold: " + thres);
			//
			//				List<Double> actReduced = new ArrayList<>(actAll);
			//				List<Double> predReduced = new ArrayList<>(predAll);
			//
			//				for (int i = test.numInstances() - 1; i >= 0; i--)
			//					if (logDensity[i] < thres)
			//					{
			//						test.remove(i);
			//						actReduced.remove(i);
			//						predReduced.remove(i);
			//					}
			//
			//				subtitle += "test-top" + (int) perc + " #i:" + test.numInstances() + "\n";
			//
			//				subtitle += "             RMSE pearson\n";
			//				subtitle += "test       " + String.format("%.2f", r1) + " " + String.format("%.2f", c1) + "\n";
			//				//			+ " "
			//				//					+ String.format("%.2f", eval.SFEntropyGain()) + " "
			//				//					+ String.format("%.2f", eval.coverageOfTestCasesByPredictedRegions()) + " "
			//				//					+ String.format("%.2f", eval.sizeOfPredictedRegions()));
			//
			//				//			if (eval.SFEntropyGain() < 0)
			//				//			{
			//				//				System.out.println("bad entropy gain");
			//				//				actReduced = null;
			//				//				predReduced = null;
			//				//			}
			//				//			else
			//				//			{
			//
			//				eval = new Evaluation(test);
			//				eval.evaluateModel(classifier, test);
			//				r2 = eval.rootMeanSquaredError();
			//				c2 = eval.correlationCoefficient();
			//				subtitle += "test-top" + (int) perc + " " + String.format("%.2f", r2) + " " + String.format("%.2f", c2)
			//						+ "\n";
			//				//			+ " "
			//				//					+ String.format("%.2f", eval.SFEntropyGain()) + " "
			//				//					+ String.format("%.2f", eval.coverageOfTestCasesByPredictedRegions()) + " "
			//				//					+ String.format("%.2f", eval.sizeOfPredictedRegions()));
			//
			//				int dr = (int) (((r2 - r1) / r1) * 100);
			//				int dc = (int) (((c2 - c1) / c1) * 100);
			//				subtitle += "% diff    " + String.format("%4d", dr) + " " + String.format("%4d", dc);
			//				String dcStr = (dc < 0 ? "-" : "") + String.format("%03d", Math.abs(dc));
			//
			//				//				if (dc < -30)
			//				plot(dcStr, new File(data).getName().replace(".arff", ""), subtitle, actAll, predAll, actReduced,
			//						predReduced);
			//				//			}
			//			}
			diffRMSE.addValue(dr);
			diffCorr.addValue(dc);
		}

	}
}
