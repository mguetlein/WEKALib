package org.mg.wekalib.appdomain;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.StandardXYBarPainter;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.function.Function2D;
import org.jfree.ui.RectangleAnchor;
import org.jfree.ui.RectangleInsets;
import org.mg.javalib.freechart.HistogramPanel;
import org.mg.javalib.util.ArrayUtil;
import org.mg.wekalib.distance.Distance;

import weka.core.Instance;
import weka.core.Instances;

public abstract class AbstractDistanceBasedADModel implements DistanceBasedADModel
{
	protected Instances trainingData;

	double[] trainingDistances;

	private double pThreshold;

	protected boolean fitToNormal = true;

	protected Distance dist;

	private double maxTrainingDistance = 0;

	protected abstract double computeDistanceInternal(int trainingIdx);

	protected abstract double computeDistanceInternal(Instance instance);

	public AbstractDistanceBasedADModel(double pThreshold, boolean fitToNormal, Distance dist)
	{
		this.pThreshold = pThreshold;
		this.fitToNormal = fitToNormal;
		this.dist = dist;
	}

	@Override
	public double computeDistance(Instance instance)
	{
		double d = computeDistanceInternal(instance);
		if (dist.getMaxDistance() == null && d > maxTrainingDistance)
		{
			System.err.println(
					"test-distance > max-train-distance!! " + d + " > " + maxTrainingDistance);
			d = maxTrainingDistance;
		}
		return d;
	}

	@Override
	public void build(Instances trainingData)
	{
		dist.build(trainingData);
		this.trainingData = trainingData;
		List<Double> trainingDist = new ArrayList<>();
		for (int i = 0; i < trainingData.size(); i++)
		{
			if (i > 0 && i % 1000 == 0)
				System.out.println(i + "/" + trainingData.size());
			double d = computeDistanceInternal(i);

			if (dist.getMaxDistance() == null)
				maxTrainingDistance = Math.max(maxTrainingDistance, d);

			if (Double.isNaN(d))
			{
				System.err.println("NaN: " + trainingData.get(i));
				continue;
			}
			trainingDist.add(d);
		}
		trainingDistances = ArrayUtil.toPrimitiveDoubleArray(trainingDist);
	}

	@Override
	public double getMaxTrainingDistance()
	{
		if (dist.getMaxDistance() != null)
			return dist.getMaxDistance();

		if (maxTrainingDistance == 0)
			throw new IllegalArgumentException("not yet determined");
		return maxTrainingDistance;
	}

	protected transient DescriptiveStatistics stats;

	protected DescriptiveStatistics getStats()
	{
		if (stats == null)
		{
			stats = new DescriptiveStatistics(trainingDistances);
			System.out.println("stats: " + stats.getMean() + " +- " + stats.getStandardDeviation());
		}
		return stats;
	}

	protected double getCumulativeProbability(Instance inst)
	{
		return getCumulativeProbability(computeDistance(inst));
	}

	protected double getInverseProbability(double pct)
	{
		if (fitToNormal)
		{
			return new NormalDistribution(getStats().getMean(), getStats().getStandardDeviation())
					.inverseCumulativeProbability(pct);
		}
		else
		{
			return getStats().getPercentile(100.0 * pct);
		}
	}

	double sorted[];

	public double getCumulativeProbability(double dist)
	{
		if (fitToNormal)
			return new NormalDistribution(getStats().getMean(), getStats().getStandardDeviation())
					.cumulativeProbability(dist);
		else
		{
			if (dist >= getStats().getMax())
				return 1.0;
			else if (dist < getStats().getMin())
				return 0.0;
			else
			{
				if (sorted == null)
					sorted = getStats().getSortedValues();
				int idx = Arrays.binarySearch(sorted, dist);
				if (idx == -1)
					throw new IllegalStateException();
				else
					return Math.abs(idx) / (double) sorted.length;
			}

		}
		//		EmpiricalDistribution emp = new EmpiricalDistribution();
		//		emp.load(trainingDistances);
		//		return emp.cumulativeProbability(computeDistance(inst));
	}

	@Override
	public boolean isInsideAppdomain(Instance testInstance)
	{
		double p = getCumulativeProbability(testInstance);
		//		System.out.println(p);
		return (p <= pThreshold);
	}

	public ChartPanel getPlot(Instance... testInstance)
	{
		List<String> labels = new ArrayList<>();
		List<double[]> vals = new ArrayList<>();

		if (testInstance != null && testInstance.length > 1)
		{
			double[] v = new double[testInstance.length];
			for (int i = 0; i < v.length; i++)
			{
				v[i] = computeDistance(testInstance[i]);
			}
			labels.add("Test instances");
			vals.add(v);
		}

		labels.add("Distance of training compounds");
		vals.add(trainingDistances);

		HistogramPanel p = new HistogramPanel("", null, "Distance", "# Training compounds", labels,
				vals, 100);

		JFreeChart chart = p.getChart();
		XYPlot plot = (XYPlot) chart.getPlot();

		//		chart.removeLegend();
		plot.setBackgroundPaint(Color.WHITE);
		plot.setRangeGridlinePaint(Color.GRAY);
		plot.setDomainGridlinePaint(Color.GRAY);
		chart.setBackgroundPaint(new Color(0, 0, 0, 0));

		XYBarRenderer render = new XYBarRenderer();
		render.setShadowVisible(false);
		StandardXYBarPainter painter = new StandardXYBarPainter();
		render.setBarPainter(painter);
		render.setSeriesPaint(0, new Color(0, 0, 0, 0));
		render.setDrawBarOutline(true);
		render.setSeriesOutlinePaint(0, Color.BLACK);
		plot.setRenderer(render);

		Color probCol = new Color(0, 139, 139);

		//		final NormalDistribution dist = new NormalDistribution(getStats().getMean(),
		//				getStats().getStandardDeviation());
		//final EmpiricalDistribution dist = new EmpiricalDistribution(4);
		//		dist.load(trainingDistances);
		Function2D func = new Function2D()
		{
			@Override
			public double getValue(double x)
			{
				return getCumulativeProbability(x);
				//return dist.probability(x - 0.01, x + 0.01);
			}
		};
		p.addFunction("Cumulative probability P(X \u2264 x)", func, probCol);

		plot.getDomainAxis().setRange(0, getInverseProbability(0.995));
		plot.getRangeAxis(1).setRange(0, plot.getRangeAxis(1).getRange().getUpperBound() * 1.1);

		double val = getInverseProbability(0.99);
		addMarker(p, val, probCol, true, "P>0.99", false, 0);
		addMarker(p, val, probCol, false, "\u21d2 Outside", false, 1);
		val = getInverseProbability(0.95);
		addMarker(p, val, probCol, true, "P\u22640.95", true, 0);
		addMarker(p, val, probCol, false, "\u21d2 Inside", true, 1);

		if (testInstance != null)
		{
			if (testInstance.length == 1)
			{
				val = computeDistance(testInstance[0]);
				addMarker(p, val, Color.RED, true, null, false, -1);
				if (val >= plot.getDomainAxis().getRange().getUpperBound())
					plot.getDomainAxis().setRange(plot.getDomainAxis().getRange().getLowerBound(),
							val * 1.1);
			}
		}

		//		if (plot.getDomainAxis().getRange().getUpperBound() > 1.05)
		//			plot.getDomainAxis().setRange(plot.getDomainAxis().getRange().getLowerBound(), 1.05);

		return p.getChartPanel();
	}

	private void addMarker(HistogramPanel p, double val, Color col, boolean drawLine, String msg,
			boolean left, int row)
	{
		XYPlot plot = (XYPlot) p.getChart().getPlot();
		Font f = plot.getDomainAxis().getTickLabelFont();
		FontMetrics fm = p.getFontMetrics(f);

		ValueMarker marker = new ValueMarker(val);
		if (msg != null)
		{
			marker.setLabel(msg);
			marker.setLabelFont(f);
			marker.setLabelPaint(col);
			marker.setLabelAnchor(left ? RectangleAnchor.TOP_LEFT : RectangleAnchor.TOP_RIGHT);
			double offset = 5 + fm.stringWidth(msg) * 0.5;
			marker.setLabelOffset(
					new RectangleInsets(15 + row * 15, left ? offset : 0, 0, left ? 0 : offset));
		}
		if (drawLine)
		{
			marker.setPaint(col);
			marker.setStroke(new BasicStroke(2.0F));
		}

		plot.addDomainMarker(marker);
	}

}
