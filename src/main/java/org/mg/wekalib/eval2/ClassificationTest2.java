package org.mg.wekalib.eval2;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import javax.swing.JPanel;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.jfree.chart.ChartColor;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYErrorRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYIntervalSeries;
import org.jfree.data.xy.XYIntervalSeriesCollection;
import org.mg.javalib.datamining.Result;
import org.mg.javalib.datamining.ResultSet;
import org.mg.javalib.datamining.ResultSetBoxPlot;
import org.mg.javalib.datamining.ResultSetFilter;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.CountedSet;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.javalib.util.ThreadUtil;
import org.mg.wekalib.eval2.data.EuclideanPCAWekaAntiStratifiedSplitter;
import org.mg.wekalib.eval2.data.TanimotoWekaAntiStratifiedSplitter;
import org.mg.wekalib.eval2.data.WekaInstancesDataSet;
import org.mg.wekalib.eval2.model.Model;
import org.mg.wekalib.eval2.model.RandomForestModel;
import org.mg.wekalib.eval2.model.RandomModel;
import org.mg.wekalib.evaluation.PredictionUtil;
import org.mg.wekalib.evaluation.PredictionUtil.ClassificationMeasure;
import org.mg.wekalib.evaluation.Predictions;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.core.Instances;

public class ClassificationTest2
{
	static ClassificationMeasure measure = ClassificationMeasure.AUC;

	static double splitRatio = 0.66;

	static boolean useTopPercentInsteadOfThreshold = false;

	enum PlotCurve
	{
		AppDomain, Confidence, Both, Nothing
	}

	static PlotCurve plotCurve = PlotCurve.AppDomain;

	static boolean boxPlot = false;

	public static boolean isNumeric(String data)
	{
		return ListUtil.createList("iris-easy", "heart-statlog", "breast-cancer", "credit-g",
				"breast-w", "diabetes", "colic.ORIG", "colic", "credit-a", "sonar", "ionosphere",
				"kr-vs-kp", "sick").contains(data);
	}

	public static void main(String[] args) throws Exception
	{
		//DB.setResultProvider(new ResultProviderImpl("/tmp/jobs/store", "/tmp/jovs/tmp"));

		//		String datasets[] = new String[] { "anneal", "anneal.ORIG", "audiology", "autos",
		//				"breast-cancer", "breast-w", "colic", "colic.ORIG", "credit-a", "credit-g",
		//				"diabetes", "glass", "heart-c", "heart-h", "heart-statlog", "hypothyroid",
		//				"ionosphere", "lymph", "primary-tumor", "segment", "sonar", "soybean", "vehicle",
		//				"vowel", "zoo", "vote", "letter", "kr-vs-kp", "sick", "CPDBAS_Mouse",
		//				"CPDBAS_Mutagenicity", "NCTRER", "AMES", "ChEMBL_8", "ChEMBL_259", "MUV_712",
		//				"MUV_644" };
		String datasets[] = new String[] { "CPDBAS_Mouse", "CPDBAS_Mutagenicity", "NCTRER", "AMES",
				"ChEMBL_8", "ChEMBL_259", "MUV_712", "MUV_644", "ChEMBL_87", "MUV_832",
				"DUD_hivrt" };
		//		String datasets[] = new String[] { "ChEMBL_8" };

		//				"ChEMBL_8", "ChEMBL_259", "MUV_712", "MUV_644" };
		//too-big: 
		//too-easy: "mushroom"
		//too-unstable: "hepatitis", "labor"
		//not-working: "iris" "balance-scale "splice" "waveform-5000"
		//ArrayUtil.scramble(datasets);

		Thread th = new Thread(new Runnable()
		{
			@Override
			public void run()
			{
				ThreadUtil.sleep(3000);

				while (true)
				{
					if (plotCurve != PlotCurve.Nothing)
						plotConfidence();

					if (boxPlot)
					{
						ResultSetBoxPlot bp = new ResultSetBoxPlot(res, "", measure + "", "Split",
								"Dataset", measure + "");
						bp.setHideMean(true);
						bp.setRotateXLabels(true);
						SwingUtil.showInFrame(bp.getChart(), measure + "", false,
								new Dimension(1200, 800));
					}
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

		for (int seed = 0; seed < 30; seed++)
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

	static ResultSet res2 = new ResultSet();

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

	static HashMap<String, String> datasetNames = new HashMap<>();

	public static String getDatasetName(String dataset)
	{
		//		if (datasetNames.containsKey(dataset))
		//			return datasetNames.get(dataset);
		//		else
		return dataset;
	}

	//	static DescriptiveStatistics v1 = new DescriptiveStatistics();
	//	static DescriptiveStatistics v2 = new DescriptiveStatistics();

	public static boolean run(String data, int seed) throws Exception
	{
		Instances inst = new Instances(new FileReader(
				System.getProperty("user.home") + "/data/weka/nominal/" + data + ".arff"));
		inst.setClassIndex(inst.numAttributes() - 1);
		//		inst.randomize(new Random(2));
		WekaInstancesDataSet ds = new WekaInstancesDataSet(inst, 0);
		System.out
				.println(data + " #inst:" + inst.numInstances() + " #feat:" + inst.numAttributes());

		//		List<String> endpoints = new ArrayList<>();
		//		for (int i = 0; i < inst.numInstances(); i++)
		//			endpoints.add(inst.get(i).stringValue(inst.classIndex()));
		//		datasetNames.put(data, data + " " + CountedSet.create(endpoints).toString());
		datasetNames.put(data, data + " i" + inst.numInstances() + " f" + inst.numAttributes());

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
			//			for (int numT : new int[] { 10, 100, 200 }) //10, 20, 40, 80, 
			//			{
			//				RandomFoktModel rf = new RandomForestModel();
			//				rf.setNumTrees(numT);
			//				classifiers.add(rf);
			//			}

			{
				classifiers.add(new RandomForestModel());
				//classifiers.add(new RandomModel());
				//classifiers.add(new NaiveBayesModel());
				//classifiers.add(new SupportVectorMachineModel());
				//				AbstainingClassifier c = new AbstainingClassifier();
				//				c.setClassifier(new RandomForest());
				//				classifiers.add(new GenericWekaModel(c));
				//				IBk knn = new IBk(3);
				//				classifiers.add(new GenericWekaModel(knn));
			}

			//			{
			//				SupportVectorMachineModel svm = new SupportVectorMachineModel();
			//				svm.setBuildLogisticModels(false);
			//				classifiers.add(svm);
			//
			//				SupportVectorMachineModel svm2 = new SupportVectorMachineModel();
			//				svm2.setBuildLogisticModels(true);
			//				classifiers.add(svm2);
			//			}

			//			{
			//				SupportVectorMachineModel svm = new SupportVectorMachineModel();
			//				svm.setBuildLogisticModels(false);
			//				svm.setKernel(new RBFKernel());
			//				classifiers.add(svm);
			//
			//				SupportVectorMachineModel svm2 = new SupportVectorMachineModel();
			//				svm2.setKernel(new RBFKernel());
			//				svm2.setBuildLogisticModels(true);
			//				classifiers.add(svm2);
			//			}

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

			//for (boolean antiStrat : new boolean[] { false, true })
			boolean antiStrat = false;
			{
				Instances instX;
				Holdout cv;
				Predictions appDomain = null;

				if (plotCurve == PlotCurve.Both || plotCurve == PlotCurve.AppDomain)
				{
					instX = new Instances(inst);
					instX.randomize(new Random(seed));
					cv = new Holdout();
					cv.setDataSet(ds);
					Model appDomainModel;
					{
						//						appDomainModel = new DistanceBasedAppDomainPredictionModel();
						appDomainModel = new RandomModel();
					}
					if (isNumeric(data))
					{
						//						if (appDomainModel instanceof DistanceBasedAppDomainPredictionModel)
						//							((DistanceBasedAppDomainPredictionModel) appDomainModel)
						//									.setNumeric(true);
						if (antiStrat)
							cv.setAntiStratifiedSplitter(
									new EuclideanPCAWekaAntiStratifiedSplitter());
					}
					else
					{
						//						if (appDomainModel instanceof DistanceBasedAppDomainPredictionModel)
						//							((DistanceBasedAppDomainPredictionModel) appDomainModel)
						//									.setNumeric(false);
						if (antiStrat)
							cv.setAntiStratifiedSplitter(new TanimotoWekaAntiStratifiedSplitter());
					}
					cv.setModel(appDomainModel);
					cv.setSplitRatio(splitRatio);
					cv.setRandomSeed(seed);
					while (!cv.isDone())
					{
						System.out.println("run ad model > ...");
						cv.nextJob().run();
						System.out.println("run ad model > done");
					}
					appDomain = cv.getResult();
				}

				//			AppDomainTest.pcaPlot(ds.getWekaInstances(), cv.getTrainingDataSet().getWekaInstances(),
				//					cv.getTestDataSet().getWekaInstances(), null);

				for (int i = 0; i < classifiers.size(); i++)
				{
					String classfierName = classifiers.get(i).getName();
					System.out.println(classfierName + " " + seed);

					instX = new Instances(inst);
					instX.randomize(new Random(seed));

					cv = new Holdout();
					if (antiStrat)
					{
						if (isNumeric(data))
							cv.setAntiStratifiedSplitter(
									new EuclideanPCAWekaAntiStratifiedSplitter());
						else
							cv.setAntiStratifiedSplitter(new TanimotoWekaAntiStratifiedSplitter());
					}
					cv.setDataSet(ds);
					cv.setModel((Model) classifiers.get(i).cloneJob());
					cv.setSplitRatio(splitRatio);
					cv.setRandomSeed(seed);
					while (!cv.isDone())
					{
						System.out.println("run pred model > ...");
						cv.nextJob().run();
						System.out.println("run pred model > done");
					}
					Predictions p = cv.getResult();

					{

						//				//				SwingUtil.showInFrame(
						//				//						PredictionUtilPlots.confidencePlot(p, ClassificationMeasure.AUC, 1.0),
						//				//						data + " confidence", false);

						//				SwingUtil.showInFrame(
						//						PredictionUtilPlots.confidencePlot(ListUtil.createList(p, pCopy),
						//								ListUtil.createList("confidence", "app-domain"),
						//								ClassificationMeasure.AUC, 1.0),
						//						data + " app domain", false);
						//				//				ScreenUtil.centerWindowsOnScreen();
						//				//				SwingUtil.waitWhileWindowsVisible();

						for (boolean conf : new boolean[] { true, false })
						{
							if (conf && plotCurve != PlotCurve.Both
									&& plotCurve != PlotCurve.Confidence)
								continue;
							if (!conf && plotCurve != PlotCurve.Both
									&& plotCurve != PlotCurve.AppDomain)
								continue;

							String m;
							if (conf)
								m = "Confidence";
							else
								m = "App-Domain";

							for (double thres : new double[] { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
									0.7, 0.8, 0.9, 1.0 })
							{
								Predictions filtered;
								double filterValues[] = conf ? p.confidence : appDomain.confidence;
								if (useTopPercentInsteadOfThreshold)
									filtered = PredictionUtil.topConfAllClasses(p, thres,
											filterValues);
								else
									filtered = PredictionUtil.filterConfAllClasses(p, thres,
											filterValues);

								if (filtered.actual.length > 0
										&& !ArrayUtil.isUniq(filtered.confidence))
								//&& ArrayUtil.getDistinctValues(filtered.actual).size() > 1)
								{
									synchronized (res2)
									{
										//										if (thres == 0.9 || thres == 0.0)
										//										{
										//											double v = PredictionUtil.getClassificationMeasure(
										//													filtered, measure, 1.0);
										//											DescriptiveStatistics d = null;
										//											if (thres == 0.0)
										//												d = v1;
										//											if (thres == 0.9)
										//												d = v2;
										//											d.addValue(v);
										//											System.err.println(thres + " " + filtered.actual.length
										//													+ " " + v + " " + d.getMean() + " +- "
										//													+ d.getStandardDeviation());
										//										}

										//										if (thres == 0)
										//										System.out.println("threshold: " + thres + " predictions: "
										//												+ filtered.actual.length + " actual:"
										//												+ CountedSet.create(
										//														ArrayUtil.toDoubleArray(filtered.actual))
										//												+ " predicted:" + CountedSet.create(ArrayUtil
										//														.toDoubleArray(filtered.predicted)));

										int idx = res2.addResult();
										res2.setResultValue(idx, "Algorithm", classfierName);
										res2.setResultValue(idx, "Measure", measure);
										res2.setResultValue(idx, "Split",
												antiStrat ? "anti-stratif." : "random");
										res2.setResultValue(idx, "Probability", m);
										res2.setResultValue(idx, "Dataset", getDatasetName(data));
										res2.setResultValue(idx, "Threshold", thres);
										res2.setResultValue(idx, "Value",
												PredictionUtil.getClassificationMeasure(filtered,
														measure, getActiveIdx(data, instX)));
									}
								}
							}
						}
					}

					if (boxPlot)
					{
						for (Predictions pf : PredictionUtil.perFold(p))
						{
							//					System.out.println(PredictionUtil.summaryClassification(pf));
							//					System.out.println(PredictionUtil.AUC(pf));
							//					System.exit(1);

							int idx = res.addResult();
							res.setResultValue(idx, "Algorithm", classfierName);
							res.setResultValue(idx, "Split",
									antiStrat ? "anti-stratif." : "random");
							res.setResultValue(idx, "Dataset", getDatasetName(data));
							res.setResultValue(idx, "Fold", pf.fold[0]);
							res.setResultValue(idx, "AUC", PredictionUtil.AUC(pf));
							res.setResultValue(idx, "AUPRC",
									PredictionUtil.AUPRC(pf, getActiveIdx(data, instX)));
							res.setResultValue(idx, ClassificationMeasure.Accuracy + "",
									PredictionUtil.accuracy(pf));

						}
					}
					//				}
				}
			}
			return true;
		}
	}

	static HashMap<String, Integer> activeIndices = new HashMap<>();

	public static Integer getActiveIdx(String data, Instances dataset)
	{
		if (!activeIndices.containsKey(data))
		{
			String classValues[] = new String[] { dataset.classAttribute().value(0),
					dataset.classAttribute().value(1) };
			Integer activeIdx = null;
			for (int i = 0; i < classValues.length; i++)
				if (classValues[i].equals("active") || classValues[i].equals("mutagen")
						|| classValues[i].equals("1") || classValues[i].equals("most-concern"))
					activeIdx = i;
			if (activeIdx == null)
				throw new IllegalStateException(
						"what is active? " + ArrayUtil.toString(classValues));
			activeIndices.put(data, activeIdx);
		}
		return activeIndices.get(data);
	}

	public static void plotConfidence()
	{
		if (res2.getNumResults() == 0)
			return;

		List<JPanel> panels = new ArrayList<>();
		List<Double> scores = new ArrayList<>();

		//HashMap<String, DescriptiveStatistics> slopeDecrCount = new HashMap<>();

		ResultSet res2Copy;
		synchronized (res2)
		{
			res2Copy = res2.copy();
		}

		for (final Object d : res2Copy.getResultValues("Dataset").values())
		{
			ResultSet res2dataset = res2Copy.filter(new ResultSetFilter()
			{
				@Override
				public boolean accept(Result result)
				{
					return result.getValue("Dataset").equals(d);
				}
			});

			List<Object> splitValues = res2dataset.getResultValues("Split").values();
			for (final Object s : splitValues)
			{
				ResultSet res2datasetSplit = res2dataset.filter(new ResultSetFilter()
				{
					@Override
					public boolean accept(Result result)
					{
						return result.getValue("Split").equals(s);
					}
				});

				String title = (splitValues.size() > 1 ? (s + " ") : "") + d;
				String yAttr = res2datasetSplit.getUniqueValue("Measure").toString();
				String xAttr;
				if (useTopPercentInsteadOfThreshold)
					xAttr = "% top predictions";
				else
					xAttr = "\u2265 Threshold";
				final XYIntervalSeriesCollection dataset = new XYIntervalSeriesCollection();

				double score = Double.NaN;

				List<String> probs = ListUtil.cast(String.class, new ArrayList<Object>(
						res2datasetSplit.getResultValues("Probability").values()));
				Collections.sort(probs);

				List<TextTitle> subtitles = new ArrayList<>();

				for (final String p : probs)
				{
					CountedSet<String> numValsPerThreshold = new CountedSet<>();

					System.out.println(p);
					if (p.toString().equals("Confidence") && plotCurve == PlotCurve.AppDomain)
						continue;
					if (p.toString().equals("App-Domain") && plotCurve == PlotCurve.Confidence)
						continue;

					ResultSet res2datasetSplitProb = res2datasetSplit.filter(new ResultSetFilter()
					{
						@Override
						public boolean accept(Result result)
						{
							return result.getValue("Probability").equals(p);
						}
					});

					HashMap<Double, DescriptiveStatistics> valueStats = new HashMap<>();
					for (int i = 0; i < res2datasetSplitProb.getNumResults(); i++)
					{
						double t = (Double) res2datasetSplitProb.getResultValue(i, "Threshold");
						double v = (Double) res2datasetSplitProb.getResultValue(i, "Value");
						if (Double.isNaN(v))
							continue;
						if (!valueStats.containsKey(t))
							valueStats.put(t, new DescriptiveStatistics());
						valueStats.get(t).addValue(v);
						numValsPerThreshold.add(t + "");
					}

					res2datasetSplitProb = res2datasetSplitProb.join(new String[] { "Algorithm",
							"Measure", "Probability", "Dataset", "Threshold" }, new String[] {},
							new String[] {});
					System.out.println(res2datasetSplitProb.toNiceString());

					SimpleRegression regr = new SimpleRegression();

					final XYIntervalSeries series = new XYIntervalSeries(p.toString());

					DescriptiveStatistics yVals = new DescriptiveStatistics();

					for (int i = 0; i < res2datasetSplitProb.getNumResults(); i++)
					{
						double t = (Double) res2datasetSplitProb.getResultValue(i, "Threshold");
						double v = (Double) res2datasetSplitProb.getResultValue(i, "Value");
						if (!Double.isNaN(v))
						{
							regr.addData(t, v);
							if (valueStats.containsKey(t))
								series.add(t, t, t, v, valueStats.get(t).getMin(),
										valueStats.get(t).getMax());
							else
								series.add(t, t, t, v, v, v);
						}
						yVals.addValue(v);
					}

					if (plotCurve != PlotCurve.Both || p.toString().equals("Confidence"))
						score = regr.getSlope() / (yVals.getMax() - yVals.getMin());

					//					//					{
					//					if (!slopeDecrCount.containsKey(p.toString()))
					//						slopeDecrCount.put(p.toString(), new DescriptiveStatistics());
					//					double slope = regr.getSlope();
					//					if (!Double.isNaN(slope))
					//						slopeDecrCount.get(p.toString()).addValue(slope);
					//					//slopeDecrCount.put(p.toString(), slopeDecrCount.get(p.toString()) + 1)
					//					//					}

					dataset.addSeries(series);
					subtitles.add(new TextTitle(numValsPerThreshold.toStringDeviationFromMax()));
				}

				final JFreeChart chart = ChartFactory.createXYLineChart(title, // chart title
						xAttr, // x axis label
						yAttr, // y axis label
						dataset, // data
						PlotOrientation.VERTICAL, true, // dataset.getSeriesCount() > 1, // include legend
						true, // tooltips
						false // urls
				);

				chart.setSubtitles(subtitles);

				chart.setBackgroundPaint(Color.WHITE);

				final XYErrorRenderer renderer = new XYErrorRenderer();

				renderer.setBaseLinesVisible(true);

				final XYPlot plot = chart.getXYPlot();
				//		renderer.setSeriesLinesVisible(0, true);
				//		renderer.setSeriesShapesVisible(0, false);//preds.length == 1);
				//			renderer.setSeriesPaint(0, FreeChartUtil.BRIGHT_RED);

				if (plotCurve == PlotCurve.Confidence)
					renderer.setSeriesPaint(0, ChartColor.LIGHT_BLUE);

				plot.setRenderer(renderer);

				//				{
				//					Function2D func = new Function2D()
				//					{
				//						@Override
				//						public double getValue(double x)
				//						{
				//							return confRegr.predict(x);
				//						}
				//					};
				//					XYDataset result = DatasetUtilities.sampleFunction2D(func, 0, 1, 300, "exp");
				//					plot.setDataset(1, result);
				//					final XYItemRenderer renderer2 = new StandardXYItemRenderer();
				//					plot.setRenderer(1, renderer2);
				//					//			final ValueAxis rangeAxis2 = new NumberAxis("exp");
				//					//			plot.setRangeAxis(1, rangeAxis2);
				//					//			plot.mapDatasetToRangeAxis(1, 1);
				//					renderer2.setSeriesPaint(0, Color.BLACK);
				//					plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD);
				//					//			rangeAxis2.setLabelFont(plot.getRangeAxis().getLabelFont());
				//					//			rangeAxis2.setTickLabelFont(plot.getRangeAxis().getTickLabelFont());
				//
				//					ValueMarker marker = new ValueMarker(0);
				//					marker.setPaint(Color.black);
				//					plot.addDomainMarker(marker);
				//					plot.addRangeMarker(marker);
				//				}

				//				plot.getDomainAxis().setAutoRange(false);
				//				plot.getRangeAxis().setAutoRange(false);
				//				plot.getDomainAxis().setRange(-0.033, 1.033);

				((NumberAxis) plot.getRangeAxis()).setAutoRangeIncludesZero(false);
				((NumberAxis) plot.getRangeAxis()).setAutoRangeStickyZero(false);
				((NumberAxis) plot.getRangeAxis()).configure();

				if (!useTopPercentInsteadOfThreshold)
					((NumberAxis) plot.getDomainAxis()).setInverted(true);

				//				plot.getRangeAxis().setRange(-0.033, 1.033);
				plot.setDomainGridlinePaint(Color.GRAY);
				plot.setRangeGridlinePaint(Color.GRAY);

				//				((NumberAxis) plot.getDomainAxis()).setTickUnit(new NumberTickUnit(0.2));
				//				((NumberAxis) plot.getRangeAxis()).setTickUnit(new NumberTickUnit(0.05));

				plot.setBackgroundAlpha(0.0F);
				//				chart.getLegend().setBackgroundPaint(new Color(0, 0, 0, 0));

				//				if (score > 0)
				chart.setBackgroundPaint(Color.WHITE);
				//				else
				//					chart.setBackgroundPaint(Color.LIGHT_GRAY);

				ChartPanel cp = new ChartPanel(chart);

				cp.setMinimumDrawHeight(200);
				cp.setMinimumDrawWidth(200);

				cp.setPreferredSize(new Dimension(300, 300));

				panels.add(cp);
				scores.add(score);
			}
		}

		//		for (String k : slopeDecrCount.keySet())
		//		{
		//			System.out.println(k + " " + slopeDecrCount.get(k).getMean() + " +- "
		//					+ slopeDecrCount.get(k).getStandardDeviation() + " (median: "
		//					+ slopeDecrCount.get(k).getPercentile(50.0) + ")");
		//		}

		int[] order = ArrayUtil.getOrdering(
				ArrayUtil.toPrimitiveDoubleArray(ListUtil.toArray(scores)),
				useTopPercentInsteadOfThreshold);

		JPanel b = new JPanel(new GridLayout(3, 5));
		b.setBackground(Color.WHITE);
		for (int i = 0; i < order.length; i++)
		{
			b.add(panels.get(order[i]));
			System.out.println(scores.get(order[i]));
		}

		String filename = "";
		if (plotCurve == PlotCurve.Both || plotCurve == PlotCurve.Confidence)
			filename += "confidence";
		if (plotCurve == PlotCurve.Both || plotCurve == PlotCurve.AppDomain)
		{
			if (!filename.isEmpty())
				filename += "-";
			filename += "app-domain";
		}
		//		SwingUtil.toFile("/home/martin/documents/app-domain/imgs/" + filename + ".png", b,
		//				b.getPreferredSize());

		SwingUtil.showInFrame(b, "confidence app-domain plots", false);
	}
}
