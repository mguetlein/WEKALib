package org.mg.wekalib.evaluation;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Shape;
import java.awt.geom.Ellipse2D;
import java.util.ArrayList;
import java.util.List;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYAnnotation;
import org.jfree.chart.annotations.XYLineAnnotation;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.chart.title.Title;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.util.ShapeUtilities;
import org.mg.javalib.freechart.FreeChartUtil;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.ListUtil;
import org.mg.javalib.util.StringUtil;
import org.mg.javalib.util.SwingUtil;
import org.mg.wekautil.Predictions;

public class PredictionsPlot
{
	String title = "";
	String subtitle = "";
	List<Predictions> preds;
	List<String> names;
	double tick = 0;

	public PredictionsPlot(Predictions... preds)
	{
		this(preds, null);
	}

	public PredictionsPlot(Predictions[] preds, String[] names)
	{
		this(ArrayUtil.toList(preds), (names != null ? ArrayUtil.toList(names) : null));
	}

	public PredictionsPlot(List<Predictions> preds, List<String> names)
	{
		this.preds = ListUtil.clone(preds);
		if (names != null)
			this.names = ListUtil.clone(names);
		else
		{
			this.names = new ArrayList<>();
			for (int i = 0; i < preds.size(); i++)
				this.names.add("");
		}
	}

	public ChartPanel getChartPanel()
	{
		DefaultXYDataset d = new DefaultXYDataset();

		for (int i = 0; i < preds.size(); i++)
		{
			names.set(i, names.get(i) + " pearson: " + StringUtil.formatDouble(PredictionUtil.pearson(preds.get(i)))
					+ " rmse: " + StringUtil.formatDouble(PredictionUtil.rmse(preds.get(i))));
			d.addSeries(names.get(i), new double[][] { preds.get(i).predicted, preds.get(i).actual });
		}

		JFreeChart f = ChartFactory.createScatterPlot(title, "predicted", "actual", d);

		List<Title> sub = new ArrayList<Title>();
		sub.add(new TextTitle(subtitle));
		sub.add(f.getLegend());
		f.setSubtitles(sub);

		XYPlot plot = f.getXYPlot();
		//					for (String k : actual.keySet())
		//					{

		XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();
		renderer.setBaseShapesFilled(false);

		//Shape cross = ShapeUtilities.createRegularCross(3, 1);
		if (preds.size() == 2)
		{
			Shape cross = ShapeUtilities.createDiagonalCross(3, 0);
			renderer.setSeriesShape(0, cross);

			//			cross = ShapeUtilities.createDiamond(4);
			cross = new Ellipse2D.Double(-3, -3, 6, 6);
			renderer.setSeriesShape(1, cross);

			//			renderer.setSeriesPaint(0, Color.BLACK);
			//			renderer.setSeriesPaint(1, Color.BLACK);

			plot.setBackgroundAlpha(0);
			plot.setDomainGridlinePaint(Color.GRAY);
			plot.setRangeGridlinePaint(Color.GRAY);
		}

		//					}
		NumberAxis yAxis = (NumberAxis) plot.getRangeAxis();
		NumberAxis xAxis = (NumberAxis) plot.getDomainAxis();
		double min = Math.min(xAxis.getRange().getLowerBound(), yAxis.getRange().getLowerBound());
		double max = Math.max(xAxis.getRange().getUpperBound(), yAxis.getRange().getUpperBound());
		xAxis.setAutoRange(false);
		yAxis.setAutoRange(false);
		yAxis.setRange(min, max);
		xAxis.setRange(min, max);

		if (tick != 0)
		{
			xAxis.setTickUnit(new NumberTickUnit(tick));
			yAxis.setTickUnit(new NumberTickUnit(tick));
		}

		XYAnnotation diagonal = new XYLineAnnotation(xAxis.getRange().getLowerBound(),
				yAxis.getRange().getLowerBound(), xAxis.getRange().getUpperBound(), yAxis.getRange().getUpperBound());
		plot.addAnnotation(diagonal);

		ChartPanel cp = new ChartPanel(f);
		cp.setMaximumDrawWidth(100000);
		cp.setMaximumDrawHeight(100000);
		cp.setPreferredSize(new Dimension(600, 600));
		return cp;
	}

	public void show(boolean wait)
	{
		ChartPanel cp = getChartPanel();
		SwingUtil.showInFrame(cp, "Plot", wait);
	}

	public void plotToPngFile(String file)
	{
		ChartPanel cp = getChartPanel();
		FreeChartUtil.toPNGFile(file, cp, cp.getPreferredSize());
	}

	public void setTitle(String title)
	{
		this.title = title;
	}

	public void setSubtitle(String subtitle)
	{
		this.subtitle = subtitle;
	}

	public void setTickUnit(double d)
	{
		this.tick = d;
	}
}
