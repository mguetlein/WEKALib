package org.mg.wekalib.evaluation;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.mg.javalib.freechart.FreeChartUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.ScreenUtil;
import org.mg.javalib.util.SwingUtil;

import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.Plot2D;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class PredictionUtilPlots extends PredictionUtil
{
	public static boolean ADD_WEKA_PLOT = false;

	public static JPanel getPlot(ClassificationMeasure m, int positiveClassValue, Predictions... p)
			throws Exception
	{
		if (m == ClassificationMeasure.AUC)
			return getROCPlot(positiveClassValue, p);
		else if (m == ClassificationMeasure.AUPRC)
			return getPRPlot(positiveClassValue, p);
		else
			throw new IllegalArgumentException();
	}

	public static JPanel getROCPlot(double positiveClassValue, Predictions... p) throws Exception
	{
		return getTCPlot(ClassificationMeasure.AUC, positiveClassValue, p);
	}

	public static JPanel getPRPlot(double positiveClassValue, Predictions... p) throws Exception
	{
		return getTCPlot(ClassificationMeasure.AUPRC, positiveClassValue, p);
	}

	protected static JPanel getTCPlot(ClassificationMeasure m, double positiveClassValue,
			Predictions... preds) throws Exception
	{
		int width = 300;
		String yAttr = "";
		String xAttr = "";
		if (m == ClassificationMeasure.AUPRC)
		{
			xAttr = "Recall";
			yAttr = "Precision";
		}
		else if (m == ClassificationMeasure.AUC)
		{
			xAttr = "False Positive Rate";
			yAttr = "Recall";//True Positive Rate";
		}

		final XYSeriesCollection dataset = new XYSeriesCollection();

		Double positiveRatio = null;

		int idx = 0;
		for (Predictions p : preds)
		{
			Instances curve = thresholdCurveInstances(p, positiveClassValue);
			//		System.err.println("X " + curve.attribute(xAttr).name());
			//		for (int i = 0; i < curve.numInstances(); i++)
			//			System.err.println(i + ": " + curve.get(i).value(curve.attribute(xAttr)));
			//		System.err.println("Y " + curve.attribute(yAttr).name());
			//		for (int i = 0; i < curve.numInstances(); i++)
			//			System.err.println(i + ": " + curve.get(i).value(curve.attribute(yAttr)));

			if (m == ClassificationMeasure.AUPRC)
			{
				double tmpPosRatio = PredictionUtil.numClassValues(p, (int) positiveClassValue)
						/ (double) p.actual.length;
				if (positiveRatio == null)
					positiveRatio = tmpPosRatio;
				else if (tmpPosRatio != positiveRatio.doubleValue())
					throw new IllegalStateException("two different baselines, not comparable!");
			}

			final XYSeries series1 = new XYSeries(m.toString() + " " + (idx++));
			//for (int i = 0; i < (2 * curve.numInstances() - 1); i++)
			for (int i = curve.numInstances() - 1; i >= 0; i--)
			{
				if (m == ClassificationMeasure.AUPRC && i == curve.numInstances() - 1)
					continue;

				//				System.err.println(curve.get(i).value(curve.attribute(xAttr)) + " "
				//						+ curve.get(i).value(curve.attribute(yAttr)));
				series1.add(curve.get(i).value(curve.attribute(xAttr)),
						curve.get(i).value(curve.attribute(yAttr)));
			}
			dataset.addSeries(series1);
		}

		XYSeries series = new XYSeries("Random");
		if (m == ClassificationMeasure.AUC)
		{
			series.add(0, 0);
			series.add(1, 1);
		}
		else
		{
			series.add(0, positiveRatio);
			series.add(1, positiveRatio);
		}
		dataset.addSeries(series);

		final JFreeChart chart = ChartFactory.createXYLineChart(null, // chart title
				xAttr, // x axis label
				yAttr, // y axis label
				dataset, // data
				PlotOrientation.VERTICAL, false, // include legend
				true, // tooltips
				false // urls
		);

		chart.setBackgroundPaint(Color.WHITE);

		final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		final XYPlot plot = chart.getXYPlot();
		for (int i = 0; i < dataset.getSeriesCount(); i++)
		{
			renderer.setSeriesLinesVisible(i, true);
			renderer.setSeriesShapesVisible(i, false);//preds.length == 1);
		}

		if (preds.length == 1)
		{
			renderer.setSeriesPaint(0, FreeChartUtil.BRIGHT_RED);
		}
		if (preds.length == 2)
		{
			renderer.setSeriesPaint(0, FreeChartUtil.BRIGHT_BLUE);
			renderer.setSeriesPaint(1, FreeChartUtil.BRIGHT_RED);
		}
		renderer.setSeriesPaint(dataset.getSeriesCount() - 1, Color.LIGHT_GRAY);

		plot.setRenderer(renderer);

		plot.getDomainAxis().setAutoRange(false);
		plot.getRangeAxis().setAutoRange(false);
		plot.getDomainAxis().setRange(-0.033, 1.033);
		plot.getRangeAxis().setRange(-0.033, 1.033);
		plot.setDomainGridlinePaint(Color.GRAY);
		plot.setRangeGridlinePaint(Color.GRAY);
		((NumberAxis) plot.getDomainAxis()).setTickUnit(new NumberTickUnit(0.25));
		((NumberAxis) plot.getRangeAxis()).setTickUnit(new NumberTickUnit(0.25));

		plot.setBackgroundPaint(Color.WHITE);

		ChartPanel cp = new ChartPanel(chart);

		cp.setMinimumDrawHeight(200);
		cp.setMinimumDrawWidth(200);

		cp.setPreferredSize(new Dimension(width, width));

		if (!ADD_WEKA_PLOT)
			return cp;
		else
		{
			JPanel res = new JPanel();
			res.add(cp);

			for (Predictions p : preds)
			{
				ThresholdVisualizePanel vmc = new ThresholdVisualizePanel()
				{
					private static final long serialVersionUID = 1L;

					@Override
					public void addPlot(PlotData2D newPlot) throws Exception
					{
						m_plot = new PlotPanel()
						{
							private static final long serialVersionUID = 1L;

							@Override
							public void addPlot(PlotData2D newPlot) throws Exception
							{
								m_plot2D = new Plot2D()
								{
									private static final long serialVersionUID = 1L;

									public void addPlot(PlotData2D newPlot) throws Exception
									{
										m_axisColour = Color.BLACK;
										super.addPlot(newPlot);
									};
								};
								m_plot2D.setBackground(Color.WHITE);
								this.add(m_plot2D, BorderLayout.CENTER);
								super.addPlot(newPlot);
							}
						};
						super.addPlot(newPlot);
					}
				};
				vmc.setROCString("(Area under ROC = "
						+ Utils.doubleToString(PredictionUtil.AUC(p), 4) + ")");
				vmc.setName("name");

				Instances curve = thresholdCurveInstances(p, positiveClassValue);

				PlotData2D tempd = new PlotData2D(curve);
				tempd.setPlotName("name2");
				tempd.addInstanceNumberAttribute();
				tempd.setCustomColour(Color.BLACK);

				// specify which points are connected
				boolean[] chartPoints = new boolean[curve.numInstances()];
				for (int n = 1; n < chartPoints.length; n++)
					chartPoints[n] = true;
				tempd.setConnectPoints(chartPoints);

				// add plot
				vmc.addPlot(tempd);

				if (m == ClassificationMeasure.AUPRC)
				{
					chartPoints[chartPoints.length - 1] = false;
				}
				else if (m == ClassificationMeasure.AUC)
				{
				}

				List<String> attributes = new ArrayList<>();
				for (int i = 0; i < curve.numAttributes(); i++)
					attributes.add(curve.attribute(i).name());
				//		System.err.println(attributes);

				vmc.setYIndex(attributes.indexOf(yAttr) + 1);
				vmc.setXIndex(attributes.indexOf(xAttr) + 1);

				//SwingUtil.showInDialog(vmc);

				JPanel panel = vmc.getPlotPanel(); // vmc;
				panel.setPreferredSize(new Dimension(width, width));
				panel.setOpaque(false);
				//		String fName = null;
				//		int i = 0;
				//		while (fName == null || new File(fName).exists())
				//			fName = "/tmp/plot/" + m + "-" + i++ + ".png";
				//		SwingUtil.toFile(fName, panel, panel.getPreferredSize());

				res.add(panel);
			}
			return res;
		}
	}

	public static JPanel confidencePlot(Predictions predictions, ClassificationMeasure m,
			double positiveClassValue)
	{
		return confidencePlot(ListUtil.createList(predictions), ListUtil.createList("Confidence"),
				m, positiveClassValue);
	}

	public static JPanel confidencePlot(List<Predictions> predictions, List<String> names,
			ClassificationMeasure m, double positiveClassValue)
	{
		int width = 300;
		String yAttr = m.toString();
		String xAttr = "% top predictions";

		final XYSeriesCollection dataset = new XYSeriesCollection();
		int idx = 0;
		for (Predictions p : predictions)
		{
			final XYSeries series = new XYSeries(names.get(idx++));
			double d = 0;
			while (true)
			{
				d = Math.min(1.0, d + 0.05);
				Predictions top = PredictionUtil.topConfAllClasses(p, d);
				System.out.println(d + " "
						+ PredictionUtil.getClassificationMeasure(top, m, positiveClassValue));
				series.add(d, PredictionUtil.getClassificationMeasure(top, m, positiveClassValue));
				if (d >= 1.0)
					break;
			}
			dataset.addSeries(series);
		}

		final JFreeChart chart = ChartFactory.createXYLineChart(null, // chart title
				xAttr, // x axis label
				yAttr, // y axis label
				dataset, // data
				PlotOrientation.VERTICAL, predictions.size() > 1, // include legend
				true, // tooltips
				false // urls
		);

		chart.setBackgroundPaint(Color.WHITE);

		final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		final XYPlot plot = chart.getXYPlot();
		//		renderer.setSeriesLinesVisible(0, true);
		//		renderer.setSeriesShapesVisible(0, false);//preds.length == 1);
		//			renderer.setSeriesPaint(0, FreeChartUtil.BRIGHT_RED);

		plot.setRenderer(renderer);

		plot.getDomainAxis().setAutoRange(false);
		plot.getRangeAxis().setAutoRange(false);
		plot.getDomainAxis().setRange(-0.033, 1.033);
		plot.getRangeAxis().setRange(-0.033, 1.033);
		plot.setDomainGridlinePaint(Color.GRAY);
		plot.setRangeGridlinePaint(Color.GRAY);
		((NumberAxis) plot.getDomainAxis()).setTickUnit(new NumberTickUnit(0.25));
		((NumberAxis) plot.getRangeAxis()).setTickUnit(new NumberTickUnit(0.25));

		plot.setBackgroundPaint(Color.WHITE);

		ChartPanel cp = new ChartPanel(chart);

		cp.setMinimumDrawHeight(200);
		cp.setMinimumDrawWidth(200);

		cp.setPreferredSize(new Dimension(width, width));

		return cp;
	}

	public static void main(String[] args) throws Exception
	{
		//		ADD_WEKA_PLOT = true;
		String s = "11011111110111000000010000000000000000000000010000000000000000100";
		//
		//		//		s = StringUtils.reverse(s);
		//		char[] sArray = s.toCharArray();
		//		//ArrayUtil.scramble(sArray);
		//		s = new String(sArray);
		//
		Predictions p = fromBitString(s);
		//		{
		//			PredictionUtil.printPredictionsWithWEKAProbability(p, 1);
		//			EnrichmentAssessment ea = PredictionUtil.toEnrichmentAssessment(p, 1);
		//			System.out.println("auc " + PredictionUtil.AUC(p) + " " + ea.auc(1.0, true));
		//
		//			System.out.println("er5 " + PredictionUtil.enrichmentFactor(p, 0.05, 1) + " "
		//					+ ea.enrichment_factor(0.05, true));
		//
		//			System.out.println("bedroc " + ea.bedroc(0.2, true));
		//		}
		//
		//		SwingUtil.showInDialog(confidencePlot(p, ClassificationMeasure.AUC, 1));

		{
			String s2 = "11001111111111000000010000000000000000000000010000000000000000100";
			Predictions p2 = fromBitString(s2);
			SwingUtil.showInFrame(PredictionUtilPlots.getROCPlot(1, p, p2), "AUC", false);
			SwingUtil.showInFrame(PredictionUtilPlots.getPRPlot(1, p, p2), "AUPRC", false);
		}

		ScreenUtil.centerWindowsOnScreen();
		SwingUtil.waitWhileWindowsVisible();
		System.exit(0);
	}

}
